# Models/mining.py

from __future__ import annotations

import numpy as np
import pandas as pd


def _month_end_close(price: pd.Series) -> pd.Series:
    """Month-end close with pandas-version fallback."""
    try:
        return price.resample("ME").last()
    except ValueError:
        return price.resample("M").last()


def _risk_on_gate_from_monthly_ma(price: pd.Series, ma_months: int = 10) -> pd.Series:
    """
    Monthly trend gate:
      risk_on = monthly_close > MA(monthly_close, ma_months)
    Forward-filled to daily index.

    This avoids within-month lookahead because the signal only updates on month-end.
    """
    monthly = _month_end_close(price)
    ma = monthly.rolling(int(ma_months), min_periods=int(ma_months)).mean()
    risk_on_m = (monthly > ma).astype(float)
    return risk_on_m.reindex(price.index, method="ffill").fillna(0.0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    minp = min(window, max(5, window // 3))
    mu = series.rolling(window, min_periods=minp).mean()
    sd = series.rolling(window, min_periods=minp).std()
    z = (series - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def mining_signal(
    df: pd.DataFrame,
    price_column: str = "IBIT_close",
    cost_column: str = "COST_TO_MINE",
    z_window: int = 180,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    use_log_edge: bool = True,
    # --- new knobs (slow regime / low turnover) ---
    smooth_span: int = 30,
    min_hold_days: int = 20,
    cooldown_days: int = 5,
    # optional trend gate
    use_trend_gate: bool = False,
    gate_ma_months: int = 10,
    allow_in_risk_on: bool = True,
) -> pd.Series:
    """
    Long-only mining-cost valuation / regime filter.

    - Edge compares price to mining cost:
        default: edge = log(price / cost)
    - Smooth edge with EMA to reduce noise.
    - Standardize with rolling z-score.

    Trade (hysteresis + low turnover):
      - Enter long when z < -entry_z
      - Exit long when z > -exit_z

    Output:
      UN-SHIFTED position in [0,1]. Engine should apply shift(1) and final clipping.
    """
    price = df[price_column].astype(float).replace([np.inf, -np.inf], np.nan)
    cost = df[cost_column].astype(float).replace([np.inf, -np.inf], np.nan)

    # Protect against bad cost values
    cost = cost.where(cost > 0)

    if use_log_edge:
        edge = np.log(price / cost)
    else:
        edge = (price - cost) / cost

    edge = edge.replace([np.inf, -np.inf], np.nan)

    # Smooth (EMA) to slow the signal
    smooth_span = int(max(2, smooth_span))
    minp = max(5, smooth_span // 3)
    edge_s = edge.ewm(span=smooth_span, min_periods=minp, adjust=False).mean()

    z = rolling_zscore(edge_s, int(z_window))

    # Optional trend gate
    if use_trend_gate:
        risk_on = _risk_on_gate_from_monthly_ma(price.dropna(), ma_months=gate_ma_months)
        risk_on = risk_on.reindex(df.index).fillna(0.0)
    else:
        risk_on = pd.Series(0.0, index=df.index)

    pos = pd.Series(0.0, index=df.index, dtype=float)
    in_pos = 0.0
    hold = 0
    cooldown = 0

    for i in range(len(df.index)):
        gated_off = (risk_on.iloc[i] > 0.5) and (not allow_in_risk_on)
        if gated_off:
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