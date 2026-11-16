"""Merkmale für den State des Agenten.

Jedes Merkmal am Tag t verwendet ausschliesslich Daten bis einschliesslich
Schlusskurs t. Die Standardisierung wird nur auf dem Trainingsabschnitt
geschätzt. Beides verhindert Look-Ahead Bias.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    logret = np.log(c).diff()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_hist = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    boll_b = (c - (ma20 - 2 * sd20)) / (4 * sd20)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean() / c

    logv = np.log(v.replace(0, np.nan))
    vol_z = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()

    feats = pd.DataFrame({
        "ret_1": logret,
        "ret_5": np.log(c).diff(5),
        "ret_20": np.log(c).diff(20),
        "volatility_20": logret.rolling(20).std(),
        "rsi_14": rsi / 100 - 0.5,
        "macd_hist": macd_hist,
        "bollinger_b": boll_b,
        "atr_14": atr,
        "volume_z": vol_z,
        "trend_50": c / c.rolling(50).mean() - 1,
    }, index=df.index)
    return feats


class Scaler:
    """Z-Standardisierung mit Kennwerten nur aus dem Trainingsabschnitt."""

    def fit(self, feats: pd.DataFrame) -> "Scaler":
        self.mean_ = feats.mean()
        self.std_ = feats.std().replace(0, 1.0)
        return self

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        return ((feats - self.mean_) / self.std_).clip(-5, 5)
