# Models/ou.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _month_end_close(price: pd.Series) -> pd.Series:
    try:
        return price.resample("ME").last()
    except ValueError:
        return price.resample("M").last()


def _risk_on_gate_from_monthly_ma(price: pd.Series, ma_months: int = 10) -> pd.Series:
    monthly = _month_end_close(price)
    ma = monthly.rolling(int(ma_months), min_periods=int(ma_months)).mean()
    risk_on_m = (monthly > ma).astype(float)
    return risk_on_m.reindex(price.index, method="ffill").fillna(0.0)


def _rolling_mean(x: pd.Series, window: int) -> pd.Series:
    w = int(window)
    minp = min(w, max(10, w // 3))
    return x.rolling(w, min_periods=minp).mean()


def _rolling_std(x: pd.Series, window: int) -> pd.Series:
    w = int(window)
    minp = min(w, max(10, w // 3))
    return x.rolling(w, min_periods=minp).std()


def ou_signal(
    price_series: pd.Series,
    # OU-ish residual
    detrend_span: int = 60,
    z_window: int = 90,
    entry_z: float = 1.0,          # dip threshold: z < -entry_z triggers overlay
    exit_z: float = 0.25,          # z > -exit_z removes overlay
    # overlay sizing
    overlay_max: float = 0.40,     # maximum additional exposure contributed by this model
    # risk filters
    use_trend_gate: bool = True,
    gate_ma_months: int = 10,
    vol_window: int = 20,
    vol_cap: float = 0.035,        # daily vol cap (3.5%); above this, no overlay
    # churn control
    min_hold_days: int = 5,
    cooldown_days: int = 3,
) -> pd.Series:
    """
    Long-only OU-style overlay on detrended log-price, UN-SHIFTED in [0,1].

    This model is NOT a full in/out strategy.
    It outputs an "overlay position" in [0, overlay_max] that you can add to a baseline.

    Logic:
      - resid = log(p) - EMA(log(p))
      - z = zscore(resid)
      - If risk-on AND vol is calm:
          enter overlay when z < -entry_z (dip)
          exit overlay when z > -exit_z
    """
    price = price_series.astype(float).replace([np.inf, -np.inf], np.nan)
    idx = price.index

    logp = np.log(price.where(price > 0)).replace([np.inf, -np.inf], np.nan)

    # detrend with EMA
    span = int(max(10, detrend_span))
    ema = logp.ewm(span=span, min_periods=max(10, span // 3), adjust=False).mean()
    resid = (logp - ema).replace([np.inf, -np.inf], np.nan)

    mu = _rolling_mean(resid, int(z_window))
    sd = _rolling_std(resid, int(z_window)).where(lambda s: s >= 1e-8)
    z = ((resid - mu) / sd).replace([np.inf, -np.inf], np.nan)

    # monthly risk-on gate
    if use_trend_gate:
        risk_on = _risk_on_gate_from_monthly_ma(price.dropna(), ma_months=gate_ma_months).reindex(idx).fillna(0.0)
    else:
        risk_on = pd.Series(1.0, index=idx)

    # realized daily vol filter (calm only)
    ret = price.pct_change()
    vol = ret.rolling(int(vol_window), min_periods=max(10, vol_window // 3)).std()
    calm = (vol <= float(vol_cap)).fillna(False)

    overlay_max = float(np.clip(overlay_max, 0.0, 1.0))

    pos = pd.Series(0.0, index=idx, dtype=float)
    in_pos = 0.0
    hold = 0
    cooldown = 0

    for i in range(len(idx)):
        allowed = (risk_on.iloc[i] > 0.5) and bool(calm.iloc[i])

        if not allowed:
            in_pos = 0.0
            hold = 0
            cooldown = 0
            pos.iloc[i] = 0.0
            continue

        if hold > 0:
            hold -= 1
        if cooldown > 0:
            cooldown -= 1

        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = in_pos
            continue

        if cooldown == 0:
            if in_pos == 0.0:
                if zi < -float(entry_z):
                    in_pos = overlay_max
                    hold = int(min_hold_days)
                    cooldown = int(cooldown_days)
            else:
                if zi > -float(exit_z):
                    in_pos = 0.0
                    hold = int(min_hold_days)
                    cooldown = int(cooldown_days)

        pos.iloc[i] = in_pos

    return pos.fillna(0.0).clip(0.0, 1.0)