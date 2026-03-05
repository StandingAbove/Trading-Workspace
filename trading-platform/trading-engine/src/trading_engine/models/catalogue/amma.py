from __future__ import annotations

import numpy as np
import pandas as pd


def amma_signal(data: pd.DataFrame, price_column: str = "IBIT_close") -> pd.Series:
    price = data[price_column].astype(float).replace([np.inf, -np.inf], np.nan)
    weights = {20: 0.25, 60: 0.25, 120: 0.25, 252: 0.25}
    pos = pd.Series(0.0, index=data.index, dtype=float)
    for window, weight in weights.items():
        mom = price.pct_change(window)
        pos += (mom > 0.0).astype(float) * weight
    return pos.fillna(0.0).clip(0.0, 1.0)


def pos_amma(df: pd.DataFrame) -> pd.Series:
    return position(df)


def position(df: pd.DataFrame) -> pd.Series:
    raw = amma_signal(df, price_column="IBIT_close")
    pos = pd.Series(raw, index=df.index).astype(float)
    pos = pos.clip(0.0, 1.0)
    return pos.shift(1).fillna(0.0)
