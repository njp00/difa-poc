"""Aufbereitung, Training und Auswertung eines Laufs."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features
from .env import TradingEnv
from .metrics import summary

SECTIONS = ("train", "validation", "test")
SECTION_LABELS = {"train": "Training", "validation": "Validierung", "test": "Test"}


def prepare(df: pd.DataFrame, cfg: dict) -> dict:
    """Merkmale berechnen, nur auf Training standardisieren, Abschnitte schneiden."""
    raw = features.compute(df)
    valid = raw.dropna().index
    raw, close = raw.loc[valid], df["close"].loc[valid]
    scaler = features.Scaler().fit(raw.loc[cfg["train"][0]:cfg["train"][1]])
    feats = scaler.transform(raw)
    window = cfg["window"]

    def section(start, end):
        # window-1 Tage Vorlauf, damit die erste Entscheidung am Starttag fällt,
        # und ein Tag Nachlauf für die Rendite des letzten Tages.
        idx = feats.index
        i0 = max(idx.searchsorted(pd.Timestamp(start)) - (window - 1), 0)
        i1 = min(idx.searchsorted(pd.Timestamp(end), side="right") + 1, len(idx))
        return feats.iloc[i0:i1], close.iloc[i0:i1]

    return {name: section(*cfg[name]) for name in SECTIONS}


def make_env(section, cfg: dict, train: bool = False) -> TradingEnv:
    f, c = section
    return TradingEnv(f, c, cfg["window"], cfg["fee"],
                      episode_length=cfg["episode_length"] if train else None)


def run_policy(env: TradingEnv, policy) -> pd.DataFrame:
    """Policy einmal deterministisch über den ganzen Abschnitt laufen lassen."""
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        obs, _, done, _, info = env.step(policy(obs))
        rows.append(info)
    return pd.DataFrame(rows).set_index("date")


def buy_and_hold(env: TradingEnv) -> pd.DataFrame:
    return run_policy(env, lambda obs: 1)


def _algo(name: str):
    from stable_baselines3 import A2C, DQN, PPO
    return {"PPO": PPO, "A2C": A2C, "DQN": DQN}[name]


def _write_status(run_dir: Path, **kw):
    path = run_dir / "status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    status.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    path.write_text(json.dumps(status), encoding="utf-8")


def train(sections: dict, cfg: dict, run_dir: Path):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    try:
        import tensorboard  # noqa: F401
        tb_dir = str(run_dir / "tensorboard")
    except ImportError:
        tb_dir = None
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "monitor").mkdir(parents=True, exist_ok=True)
    seeds, total = cfg["seeds"], cfg["timesteps"]

    class Progress(BaseCallback):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def _on_step(self):
            if self.num_timesteps % 2048 == 0 or self.num_timesteps == total:
                _write_status(run_dir, state="training", seed=seeds[self.i],
                              progress=(self.i + self.num_timesteps / total) / len(seeds))
            return True

    Algo = _algo(cfg["algorithm"])
    for i, seed in enumerate(seeds):
        env = Monitor(make_env(sections["train"], cfg, train=True),
                      filename=str(run_dir / "monitor" / f"seed{seed}"))
        model = Algo("MlpPolicy", env, seed=seed, verbose=0, tensorboard_log=tb_dir)
        model.learn(total_timesteps=total, tb_log_name=f"{cfg['algorithm'].lower()}_seed{seed}",
                    callback=Progress(i))
        model.save(run_dir / "models" / f"seed{seed}")
        print(f"Seed {seed}: Training abgeschlossen", flush=True)


def _model_path(run_dir: Path, cfg: dict, seed: int) -> Path:
    for name in (f"seed{seed}.zip", f"{cfg['algorithm'].lower()}_seed{seed}.zip"):
        if (run_dir / "models" / name).exists():
            return run_dir / "models" / name
    raise FileNotFoundError(f"Kein Modell für Seed {seed} in {run_dir / 'models'}")


def evaluate(sections: dict, cfg: dict, run_dir: Path, section_name: str) -> pd.DataFrame:
    Algo = _algo(cfg["algorithm"])
    env = make_env(sections[section_name], cfg)
    algo = cfg["algorithm"]
    results, curves = {}, {}
    bh = buy_and_hold(env)
    bh.to_csv(run_dir / f"{section_name}_buy_and_hold.csv")
    results["Buy-and-Hold"] = summary(bh["return"], cfg["annualization"], bh["position"])
    curves["Buy-and-Hold"] = bh["value"]
    for seed in cfg["seeds"]:
        model = Algo.load(_model_path(run_dir, cfg, seed))
        log = run_policy(env, lambda obs: int(model.predict(obs, deterministic=True)[0]))
        results[f"{algo} Seed {seed}"] = summary(log["return"], cfg["annualization"], log["position"])
        curves[f"{algo} Seed {seed}"] = log["value"]
        log.to_csv(run_dir / f"{section_name}_seed{seed}.csv")
    table = pd.DataFrame(results).T
    agents = table.loc[[k for k in table.index if k != "Buy-and-Hold"]]
    table.loc[f"{algo} Mittelwert"] = agents.mean()
    table.loc[f"{algo} Standardabweichung"] = agents.std()
    table.to_csv(run_dir / f"{section_name}_kennzahlen.csv")
    _plot(pd.DataFrame(curves), section_name, run_dir, algo)
    return table


def random_benchmark(env_returns: np.ndarray, mean_holding: float, fee: float,
                     n: int = 5000, seed: int = 0) -> np.ndarray:
    """Endwerte zufälliger Strategien mit gleicher mittlerer Haltedauer wie der Agent."""
    rng = np.random.default_rng(seed)
    p = 1 / max(mean_holding, 1.0)
    out = np.empty(n)
    for k in range(n):
        switches = rng.random(len(env_returns)) < p
        pos = np.cumsum(switches) % 2
        if rng.random() < 0.5:
            pos = 1 - pos
        cost = fee * np.abs(np.diff(np.r_[0, pos]))
        out[k] = np.prod((1 - cost) * (1 + pos * env_returns))
    return out


def mean_holding_period(positions: pd.Series) -> float:
    runs = (positions != positions.shift()).cumsum()
    return float(positions.groupby(runs).size().mean())


def _plot(curves: pd.DataFrame, section_name: str, run_dir: Path, algo: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curves.index, curves["Buy-and-Hold"], color="black", lw=2, label="Buy-and-Hold")
    for col in curves.columns.drop("Buy-and-Hold"):
        ax.plot(curves.index, curves[col], lw=1, alpha=0.8, label=col)
    ax.set_yscale("log")
    ax.set_ylabel("Portfoliowert (Start = 1, log. Skala)")
    label = SECTION_LABELS.get(section_name, section_name)
    ax.set_title(f"BTC/USDT, {label}: {algo} gegen Buy-and-Hold (nach Kosten)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / f"{section_name}_equity.png", dpi=150)
    plt.close(fig)
