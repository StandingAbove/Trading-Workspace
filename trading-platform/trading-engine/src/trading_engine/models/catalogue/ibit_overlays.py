import numpy as np
import pandas as pd


def compute_overlay(px: pd.Series) -> pd.Series:
    px = px.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    idx = px.index

    sma200 = px.rolling(200, min_periods=200).mean()
    below_200 = px < sma200

    dd = px / px.cummax() - 1.0
    ret20 = px.pct_change(20)

    crash_on = (dd < -0.20) & below_200 & (ret20 < -0.08)
    crash_off = (~below_200) | (ret20 > 0.05)

    crash = pd.Series(False, index=idx)
    state = False
    for t in idx:
        if (not state) and bool(crash_on.loc[t]):
            state = True
        elif state and bool(crash_off.loc[t]):
            state = False
        crash.loc[t] = state

    # crash multiplier
    overlay = pd.Series(1.0, index=idx)
    overlay.loc[crash] = 0.15

    # volatility multiplier (panic vol => reduce)
    vol20 = px.pct_change().rolling(20).std()
    vol_ref = vol20.rolling(200, min_periods=50).median()
    high_vol = vol20 > (2.0 * vol_ref)
    overlay.loc[high_vol] *= 0.6

    # trend slope multiplier (weak long trend => reduce)
    slope = sma200.diff(20)
    weak_trend = slope < 0
    overlay.loc[weak_trend] *= 0.8

    overlay = overlay.clip(0.0, 1.0)

    # trade next bar
    overlay = overlay.shift(1).fillna(0.0)
    overlay.name = "overlay"
    return overlay
