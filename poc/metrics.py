"""Kennzahlen zur Bewertung einer Renditereihe (Tagesrenditen)."""
import numpy as np
import pandas as pd


def summary(returns: pd.Series, periods: int, positions: pd.Series | None = None) -> dict:
    r = returns.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / periods
    ann_ret = equity.iloc[-1] ** (1 / years) - 1
    ann_vol = r.std() * np.sqrt(periods)
    downside = r[r < 0].std() * np.sqrt(periods)
    drawdown = equity / equity.cummax() - 1
    out = {
        "Gesamtrendite": equity.iloc[-1] - 1,
        "Rendite p.a.": ann_ret,
        "Volatilität p.a.": ann_vol,
        "Sharpe Ratio": r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan,
        "Sortino Ratio": r.mean() * periods / downside if downside > 0 else np.nan,
        "Max Drawdown": drawdown.min(),
    }
    if positions is not None:
        start = pd.Series([0.0])  # jede Strategie beginnt ohne Position
        out["Positionswechsel"] = int((pd.concat([start, positions.reset_index(drop=True)]).diff().abs() > 1e-9).sum())
        out["Investitionsgrad"] = positions.mean()
    return out
