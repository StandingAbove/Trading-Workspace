import numpy as np
import pandas as pd


def _rolling_mean(x: pd.Series, window: int) -> pd.Series:
    window = int(window)
    minp = min(window, max(3, window // 3))
    return x.rolling(window, min_periods=minp).mean()


def _rolling_std(x: pd.Series, window: int) -> pd.Series:
    window = int(window)
    minp = min(window, max(3, window // 3))
    return x.rolling(window, min_periods=minp).std()


def zscore_signal(
    price_series: pd.Series,
    window: int = 180,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    long_short: bool = False,     # kept for compatibility; we enforce long-only anyway
    max_leverage: float = 1.0,    # hard-capped to <= 1.0
    vol_window: int = 30,
    vol_target: float = 0.02,
) -> pd.Series:
    """
    Long-only mean reversion on rolling z-score of price.

    Logic:
      z = (price - mean) / std
      - Enter long when z < -entry_z
      - Exit when z > -exit_z

    Output is UN-SHIFTED target position in [0, 1].
    """
    price = price_series.astype(float).replace([np.inf, -np.inf], np.nan)

    mu = _rolling_mean(price, window)
    sd = _rolling_std(price, window)

    z = (price - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan)

    pos = pd.Series(0.0, index=price.index)
    in_pos = 0.0

    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = in_pos
            continue

        if in_pos == 0.0:
            if zi < -float(entry_z):
                in_pos = 1.0
        else:
            if zi > -float(exit_z):
                in_pos = 0.0

        pos.iloc[i] = in_pos

    # Optional: scale DOWN in high vol (never scale up)
    vol_window = int(vol_window)
    if vol_window > 1 and vol_target is not None and np.isfinite(vol_target) and vol_target > 0:
        ret = price.pct_change()
        vol = ret.rolling(vol_window, min_periods=max(3, vol_window // 3)).std()
        scale = (vol_target / vol).replace([np.inf, -np.inf], np.nan)
        scale = scale.clip(0.0, 1.0)  # never lever up
        pos = (pos * scale).fillna(0.0)

    pos = pos.fillna(0.0)
    pos = pos.clip(0.0, float(min(max_leverage, 1.0)))
    return pos