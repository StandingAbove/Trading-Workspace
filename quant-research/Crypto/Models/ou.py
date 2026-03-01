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
    minp = min(w, max(5, w // 3))
    return x.rolling(w, min_periods=minp).mean()


def _rolling_var(x: pd.Series, window: int) -> pd.Series:
    w = int(window)
    minp = min(w, max(5, w // 3))
    return x.rolling(w, min_periods=minp).var()


def _rolling_std(x: pd.Series, window: int) -> pd.Series:
    w = int(window)
    minp = min(w, max(5, w // 3))
    return x.rolling(w, min_periods=minp).std()


def _lag1_no_shift(x: pd.Series) -> pd.Series:
    """Lag by 1 without calling .shift()."""
    arr = x.to_numpy(dtype=float)
    lag = np.empty_like(arr)
    lag[0] = np.nan
    lag[1:] = arr[:-1]
    return pd.Series(lag, index=x.index)


def ou_signal(
    price_series: pd.Series,
    window: int = 120,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    long_short: bool = False,  # kept for compatibility; enforced long-only
    detrend_window: int = 120,
    # regime gate
    use_trend_gate: bool = True,
    gate_ma_months: int = 10,
    allow_in_risk_on: bool = False,     # default: trade only when risk_off/sideways
    allow_in_risk_off: bool = True,
    # churn control
    min_hold_days: int = 5,
    cooldown_days: int = 3,
    # OU-quality filters (default OFF so it trades)
    require_mean_reversion: bool = False,
    min_half_life: int = 5,
    max_half_life: int = 365,
) -> pd.Series:
    """
    Long-only OU-style mean reversion on detrended log-price.

    Uses OU z if available; otherwise falls back to a simple z-score on the detrended series
    so the model doesn't die (flatline) from NaNs.

    Output is UN-SHIFTED position in [0,1].
    """
    price = price_series.astype(float).replace([np.inf, -np.inf], np.nan)
    idx = price.index

    logp = np.log(price.where(price > 0)).replace([np.inf, -np.inf], np.nan)

    # detrend
    mu_trend = _rolling_mean(logp, detrend_window)
    x = (logp - mu_trend).replace([np.inf, -np.inf], np.nan)

    # lag without .shift()
    x_lag = _lag1_no_shift(x)

    # rolling AR(1) via moments
    w = int(window)
    ex = _rolling_mean(x_lag, w)
    ey = _rolling_mean(x, w)
    exy = _rolling_mean(x_lag * x, w)
    ex2 = _rolling_mean(x_lag * x_lag, w)

    cov_xy = exy - ex * ey
    var_x = (ex2 - ex * ex).replace([np.inf, -np.inf], np.nan)
    var_x = var_x.where(var_x.abs() > 1e-12)

    b = (cov_xy / var_x).replace([np.inf, -np.inf], np.nan).clip(-0.999, 0.999)
    a = (ey - b * ex).replace([np.inf, -np.inf], np.nan)

    denom = (1.0 - b).replace(0.0, np.nan)
    mu_hat = (a / denom).replace([np.inf, -np.inf], np.nan)

    eps = (x - (a + b * x_lag)).replace([np.inf, -np.inf], np.nan)
    var_eps = _rolling_var(eps, w).replace([np.inf, -np.inf], np.nan)

    denom_var = (1.0 - b * b).replace(0.0, np.nan)
    var_stat = (var_eps / denom_var).replace([np.inf, -np.inf], np.nan)
    sd_stat = np.sqrt(var_stat).where(lambda s: s >= 1e-8)

    z_ou = ((x - mu_hat) / sd_stat).replace([np.inf, -np.inf], np.nan)

    # fallback z-score so OU never becomes "always NaN"
    mu_x = _rolling_mean(x, w)
    sd_x = _rolling_std(x, w).where(lambda s: s >= 1e-8)
    z_fb = ((x - mu_x) / sd_x).replace([np.inf, -np.inf], np.nan)

    z = z_ou.where(z_ou.notna(), z_fb)

    # regime gate
    if use_trend_gate:
        risk_on = _risk_on_gate_from_monthly_ma(price.dropna(), ma_months=gate_ma_months)
        risk_on = risk_on.reindex(idx).fillna(0.0)
    else:
        risk_on = pd.Series(0.0, index=idx)

    # optional mean reversion filter
    if require_mean_reversion:
        b_pos = b.where((b > 0.0) & (b < 0.999))
        kappa = -np.log(b_pos)
        half_life = np.log(2.0) / kappa
        ok = half_life.between(float(min_half_life), float(max_half_life)).fillna(False)
    else:
        ok = pd.Series(True, index=idx)

    pos = pd.Series(0.0, index=idx, dtype=float)
    in_pos = 0.0
    hold = 0
    cooldown = 0

    for i in range(len(idx)):
        is_risk_on = (risk_on.iloc[i] > 0.5)
        allowed = (is_risk_on and allow_in_risk_on) or ((not is_risk_on) and allow_in_risk_off)
        if not allowed:
            in_pos = 0.0
            hold = 0
            cooldown = 0
            pos.iloc[i] = 0.0
            continue

        if not bool(ok.iloc[i]):
            # don't allow entries when fit looks bad; force flat
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
                    in_pos = 1.0
                    hold = int(min_hold_days)
                    cooldown = int(cooldown_days)
            else:
                if zi > -float(exit_z):
                    in_pos = 0.0
                    hold = int(min_hold_days)
                    cooldown = int(cooldown_days)

        pos.iloc[i] = in_pos

    return pos.fillna(0.0).clip(0.0, 1.0)