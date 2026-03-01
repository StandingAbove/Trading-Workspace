# Models/zscore.py

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


def _rolling_std(x: pd.Series, window: int) -> pd.Series:
    w = int(window)
    minp = min(w, max(5, w // 3))
    return x.rolling(w, min_periods=minp).std()


def zscore_signal(
    price_series: pd.Series,
    window: int = 90,
    entry_z: float = 1.75,
    exit_z: float = 0.25,
    long_short: bool = False,
    max_leverage: float = 1.0,
    vol_window: int = 30,
    vol_target: float | None = None,
    # trend gate
    use_trend_gate: bool = True,
    gate_ma_months: int = 10,
    allow_in_risk_on: bool = True,
    allow_in_risk_off: bool = False,
    # residual detrending
    detrend_ema_span: int = 60,
    # churn control
    min_hold_days: int = 5,
    cooldown_days: int = 3,
    # NEW: behavior switch
    mode: str = "derisk_overbought",  # "dip_buy" or "derisk_overbought"
) -> pd.Series:
    """
    Long-only zscore model on residual:
      resid = log(price) - EMA(log(price))

    mode="dip_buy" (old behavior):
      - Enter long when z < -entry_z
      - Exit when z > -exit_z

    mode="derisk_overbought" (recommended for ensembles):
      - Base is invested (1) when allowed (risk-on by default)
      - De-risk to 0 when z > entry_z
      - Re-enter to 1 when z < exit_z

    Output: UN-SHIFTED position in [0,1].
    """
    price = price_series.astype(float).replace([np.inf, -np.inf], np.nan)
    idx = price.index

    logp = np.log(price.where(price > 0)).replace([np.inf, -np.inf], np.nan)

    ema = logp.ewm(
        span=int(detrend_ema_span),
        min_periods=max(5, detrend_ema_span // 3),
        adjust=False,
    ).mean()
    resid = (logp - ema).replace([np.inf, -np.inf], np.nan)

    mu = _rolling_mean(resid, window)
    sd = _rolling_std(resid, window).where(lambda s: s >= 1e-8)
    z = ((resid - mu) / sd).replace([np.inf, -np.inf], np.nan)

    if use_trend_gate:
        risk_on = _risk_on_gate_from_monthly_ma(price.dropna(), ma_months=gate_ma_months)
        risk_on = risk_on.reindex(idx).fillna(0.0)
    else:
        risk_on = pd.Series(1.0, index=idx)

    pos = pd.Series(0.0, index=idx, dtype=float)

    in_pos = 0.0
    hold = 0
    cooldown = 0
    prev_allowed = False

    for i in range(len(idx)):
        is_risk_on = (risk_on.iloc[i] > 0.5)
        allowed = (is_risk_on and allow_in_risk_on) or ((not is_risk_on) and allow_in_risk_off)

        if not allowed:
            in_pos = 0.0
            hold = 0
            cooldown = 0
            prev_allowed = False
            pos.iloc[i] = 0.0
            continue

        # entering an allowed regime: default invested for derisk mode
        if (not prev_allowed) and (mode == "derisk_overbought"):
            in_pos = 1.0
        prev_allowed = True

        if hold > 0:
            hold -= 1
        if cooldown > 0:
            cooldown -= 1

        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = in_pos
            continue

        if cooldown == 0:
            if mode == "dip_buy":
                # Buy dips
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

            elif mode == "derisk_overbought":
                # Sell rips / reduce exposure when stretched
                if in_pos == 1.0:
                    if zi > float(entry_z):
                        in_pos = 0.0
                        hold = int(min_hold_days)
                        cooldown = int(cooldown_days)
                else:
                    if zi < float(exit_z):
                        in_pos = 1.0
                        hold = int(min_hold_days)
                        cooldown = int(cooldown_days)
            else:
                raise ValueError("mode must be 'dip_buy' or 'derisk_overbought'")

        pos.iloc[i] = in_pos

    # Optional vol scaling DOWN only
    if vol_target is not None and np.isfinite(vol_target) and float(vol_target) > 0 and int(vol_window) > 1:
        ret = price.pct_change()
        vol = ret.rolling(int(vol_window), min_periods=max(5, vol_window // 3)).std()
        scale = (float(vol_target) / vol).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
        pos = (pos * scale).fillna(0.0)

    cap = float(min(max_leverage, 1.0))
    return pos.fillna(0.0).clip(0.0, cap)