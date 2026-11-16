"""Download der Bitcoin-Tagesdaten von Binance Vision.

Binance stellt historische Kursdaten als monatliche ZIP-Dateien frei zur
Verfügung (https://data.binance.vision). Die Dateien werden lokal
zwischengespeichert, damit jeder Lauf auf identischen Rohdaten beruht.
"""
import io
import urllib.request
import zipfile

import pandas as pd

from . import config

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/{i}/{s}-{i}-{m}.zip"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _months(first: str, last: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(first, last, freq="M")]


def _to_datetime(ms_or_us: pd.Series) -> pd.Series:
    # Seit 2025 liefert Binance Zeitstempel in Mikrosekunden statt Millisekunden.
    unit = "us" if ms_or_us.iloc[0] > 10**14 else "ms"
    return pd.to_datetime(ms_or_us, unit=unit, utc=True)


def download(force: bool = False) -> pd.DataFrame:
    raw_dir = config.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for month in _months(config.FIRST_MONTH, config.LAST_MONTH):
        path = raw_dir / f"{config.SYMBOL}-{config.INTERVAL}-{month}.zip"
        if force or not path.exists():
            url = BASE_URL.format(s=config.SYMBOL, i=config.INTERVAL, m=month)
            print(f"Lade {url}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                path.write_bytes(resp.read())
        with zipfile.ZipFile(path) as zf:
            with zf.open(zf.namelist()[0]) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh), header=None)
        if not str(df.iloc[0, 0]).isdigit():   # einige Dateien enthalten eine Kopfzeile
            df = df.iloc[1:]
        df.columns = COLUMNS
        df["open_time"] = df["open_time"].astype("int64")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    parts = []
    for _, part in df.groupby(df["open_time"] > 10**14):
        part = part.copy()
        part["date"] = _to_datetime(part["open_time"]).dt.tz_convert(None).dt.normalize()
        parts.append(part)
    df = pd.concat(parts).set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")]
    out = config.DATA_DIR / f"{config.SYMBOL.lower()}_{config.INTERVAL}.csv"
    df.to_csv(out)
    print(f"{len(df)} Tage gespeichert in {out}")
    return df


def load() -> pd.DataFrame:
    path = config.DATA_DIR / f"{config.SYMBOL.lower()}_{config.INTERVAL}.csv"
    if not path.exists():
        return download()
    return pd.read_csv(path, index_col="date", parse_dates=True)


def check(df: pd.DataFrame) -> None:
    """Einfache Qualitätsprüfung: Lücken und unplausible Werte melden."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full.difference(df.index)
    print(f"Zeitraum {df.index.min().date()} bis {df.index.max().date()}, {len(df)} Tage, "
          f"{len(missing)} fehlende Tage")
    if len(missing):
        print("Fehlend:", ", ".join(str(d.date()) for d in missing[:10]))
    bad = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])]
    if len(bad):
        print(f"Warnung: {len(bad)} unplausible Zeilen")
