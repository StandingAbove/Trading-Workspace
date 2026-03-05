import numpy as np
import pandas as pd


def trend_signal(df: pd.DataFrame, price_column: str = "IBIT_close") -> pd.Series:
    price = df[price_column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    monthly_close = price.resample("ME").last()
    ma = monthly_close.rolling(10, min_periods=10).mean()
    sig_m = (monthly_close > ma).astype(float)
    sig_d = sig_m.reindex(price.index, method="ffill").fillna(0.0)
    return sig_d.reindex(df.index).fillna(0.0).clip(0.0, 1.0)


def pos_trend(df: pd.DataFrame) -> pd.Series:
    return position(df)


def position(df: pd.DataFrame) -> pd.Series:
    raw = trend_signal(df, price_column="IBIT_close")
    pos = pd.Series(raw, index=df.index).astype(float)
    pos = pos.clip(0.0, 1.0)
    return pos.shift(1).fillna(0.0)
