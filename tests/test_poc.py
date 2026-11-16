"""Prüfungen gegen die häufigsten Fehlerquellen."""
import numpy as np
import pandas as pd
import pytest

from poc import features
from poc.env import TradingEnv
from poc.metrics import summary


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 10000 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98,
                         "close": close, "volume": rng.uniform(1e3, 2e3, n)}, index=idx)


def test_features_have_no_lookahead():
    """Merkmale bis Tag t dürfen sich nicht ändern, wenn spätere Daten fehlen."""
    df = synthetic()
    full = features.compute(df)
    cut = features.compute(df.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], cut)


def test_env_reward_uses_only_next_day():
    df = synthetic()
    f = features.compute(df).dropna()
    c = df["close"].loc[f.index]
    env = TradingEnv(f, c, window=10, fee=0.0)
    env.reset()
    t = env.t
    _, r, _, _, _ = env.step(1)
    assert np.isclose(r, np.log(c.iloc[t + 1] / c.iloc[t]))


def test_costs_and_flat_position():
    df = synthetic()
    f = features.compute(df).dropna()
    c = df["close"].loc[f.index]
    env = TradingEnv(f, c, window=10, fee=0.001)
    env.reset()
    _, r_buy, *_ = env.step(1)
    _, r_sell, _, _, info = env.step(0)
    assert np.isclose(r_sell, np.log(1 - 0.001))      # Verkauf kostet nur Gebühr
    assert info["position"] == 0


def test_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    df = synthetic()
    f = features.compute(df).dropna()
    env = TradingEnv(f, df["close"].loc[f.index], window=10, fee=0.001, episode_length=50)
    check_env(env, skip_render_check=True)


def test_buy_and_hold_metrics():
    r = pd.Series([0.01] * 365)
    m = summary(r, 365)
    assert m["Max Drawdown"] == 0
    assert m["Gesamtrendite"] == pytest.approx(1.01 ** 365 - 1)


def test_buy_and_hold_counts_one_trade():
    r = pd.Series([0.01] * 10)
    m = summary(r, 365, pd.Series([1.0] * 10))
    assert m["Positionswechsel"] == 1


def test_binance_parsing_ms_us_and_header(tmp_path, monkeypatch):
    """Binance-Dateien: Millisekunden bis 2024, Mikrosekunden ab 2025, teils mit Kopfzeile."""
    import io, zipfile
    from poc import config, data

    def fake_zip(month):
        ts = pd.Timestamp(month + "-01")
        factor = 1000 if ts.year >= 2025 else 1
        rows = []
        for d in range(2):
            t = int((ts + pd.Timedelta(days=d)).value // 10**6) * factor
            rows.append(f"{t},1,2,0.5,1.5,10,{t},0,0,0,0,0")
        text = ("open_time,open,high,low,close,volume,close_time,a,b,c,d,e\n" if month == "2025-01" else "") + "\n".join(rows)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("x.csv", text)
        return buf.getvalue()

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "FIRST_MONTH", "2024-12")
    monkeypatch.setattr(config, "LAST_MONTH", "2025-01")
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url, timeout: Resp(fake_zip(url[-11:-4])))
    df = data.download()
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-12-01", "2024-12-02", "2025-01-01", "2025-01-02"]
    assert (df["close"] == 1.5).all()


def test_pipeline_end_to_end(tmp_path):
    """Ganzer Ablauf mit Testdaten: Training, Auswertung, Zufallsbenchmark."""
    from poc import config, pipeline
    idx = pd.date_range("2017-08-17", "2025-12-31", freq="D")
    rng = np.random.default_rng(1)
    close = 4000 * np.exp(np.cumsum(rng.normal(0.001, 0.03, len(idx))))
    df = pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98,
                       "close": close, "volume": rng.uniform(1e4, 5e4, len(idx))}, index=idx)
    cfg = dict(config.DEFAULTS, seeds=[0], timesteps=512)
    sec = pipeline.prepare(df, cfg)
    pipeline.train(sec, cfg, tmp_path)
    table = pipeline.evaluate(sec, cfg, tmp_path, "test")
    assert "Buy-and-Hold" in table.index and (tmp_path / "test_seed0.csv").exists()
    log = pd.read_csv(tmp_path / "test_seed0.csv", index_col=0, parse_dates=True)
    assert log.index[0] == pd.Timestamp(cfg["test"][0])
    r = pd.read_csv(tmp_path / "test_buy_and_hold.csv", index_col=0)["return"].to_numpy()
    sims = pipeline.random_benchmark(r, 3.0, cfg["fee"], n=50)
    assert sims.shape == (50,) and (sims > 0).all()
