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


def ou_signal(
    price_series: pd.Series,
    window: int = 180,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
    long_short: bool = False,     # kept for compatibility; we enforce long-only anyway
    detrend_window: int = 180,
) -> pd.Series:
    """
    Long-only OU-style mean reversion on detrended log-price.

    Steps:
      - logp = log(price)
      - detrend with rolling mean
      - z = (logp - mean) / std

    Trade:
      - Enter long when z < -entry_z
      - Exit when z > -exit_z

    Output is UN-SHIFTED target position in [0, 1].
    """
    price = price_series.astype(float).replace(0, np.nan)
    logp = np.log(price).replace([np.inf, -np.inf], np.nan)

    detrend_window = int(detrend_window)
    mu = _rolling_mean(logp, detrend_window)
    resid = logp - mu

    sd = _rolling_std(resid, window)
    z = (resid / sd).replace([np.inf, -np.inf], np.nan)

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

    return pos.fillna(0.0).clip(0.0, 1.0)