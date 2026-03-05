from __future__ import annotations

import numpy as np
import pandas as pd


def mining_signal(df: pd.DataFrame, price_column: str = "IBIT_close", cost_column: str = "COST_TO_MINE") -> pd.Series:
    idx = df.index
    price = df[price_column].astype(float).replace([np.inf, -np.inf], np.nan)
    cost = df.get(cost_column, pd.Series(index=idx, data=np.nan)).astype(float).replace([np.inf, -np.inf], np.nan)
    cost = cost.where(cost > 0)
    edge = np.log(price / cost).replace([np.inf, -np.inf], np.nan)
    edge_s = edge.ewm(span=45, min_periods=15, adjust=False).mean()
    mu = edge_s.rolling(252, min_periods=84).mean()
    sd = edge_s.rolling(252, min_periods=84).std().where(lambda s: s >= 1e-8)
    z = ((edge_s - mu) / sd).replace([np.inf, -np.inf], np.nan)
    t = (z + 1.0) / 2.0
    expo = 1.0 + (0.35 - 1.0) * t
    return expo.clip(lower=0.35, upper=1.0).fillna(0.35).reindex(idx).fillna(0.35)


def pos_mining(df: pd.DataFrame) -> pd.Series:
    return position(df)


def position(df: pd.DataFrame) -> pd.Series:
    raw = mining_signal(df)
    pos = pd.Series(raw, index=df.index).astype(float)
    pos = pos.clip(0.0, 1.0)
    return pos.shift(1).fillna(0.0)
