import numpy as np
import pandas as pd


def zscore_signal(price_series: pd.Series) -> pd.Series:
    price = price_series.astype(float).replace([np.inf, -np.inf], np.nan)
    logp = np.log(price.where(price > 0)).replace([np.inf, -np.inf], np.nan)
    ema = logp.ewm(span=60, min_periods=20, adjust=False).mean()
    resid = (logp - ema).replace([np.inf, -np.inf], np.nan)
    mu = resid.rolling(90, min_periods=30).mean()
    sd = resid.rolling(90, min_periods=30).std().where(lambda s: s >= 1e-8)
    z = ((resid - mu) / sd).replace([np.inf, -np.inf], np.nan)

    pos = pd.Series(0.0, index=price.index, dtype=float)
    in_pos = 1.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            pos.iloc[i] = in_pos
            continue
        if in_pos == 1.0 and zi > 1.75:
            in_pos = 0.0
        elif in_pos == 0.0 and zi < 0.25:
            in_pos = 1.0
        pos.iloc[i] = in_pos

    return pos.fillna(0.0).clip(0.0, 1.0)


def pos_zscore(df: pd.DataFrame) -> pd.Series:
    return position(df)


def position(df: pd.DataFrame) -> pd.Series:
    raw = zscore_signal(df["IBIT_close"])
    pos = pd.Series(raw, index=df.index).astype(float)
    pos = pos.clip(0.0, 1.0)
    return pos.shift(1).fillna(0.0)
