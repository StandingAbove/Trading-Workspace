# Models/mining.py
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


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    w = int(window)
    minp = min(w, max(10, w // 3))
    mu = x.rolling(w, min_periods=minp).mean()
    sd = x.rolling(w, min_periods=minp).std()
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def mining_signal(
    df: pd.DataFrame,
    price_column: str = "IBIT_close",
    cost_column: str = "COST_TO_MINE",
    # edge + normalization
    z_window: int = 252,
    use_log_edge: bool = True,
    # slow the signal down
    smooth_span: int = 45,
    # convert z -> exposure
    z_lo: float = -1.0,      # "cheap" boundary
    z_hi: float = +1.0,      # "expensive" boundary
    min_exposure: float = 0.35,
    max_exposure: float = 1.00,
    # optional trend regime gate (monthly)
    use_trend_gate: bool = True,
    gate_ma_months: int = 10,
    risk_off_cap: float = 0.60,  # if risk-off, cap exposure here (still avoids full cash drag)
) -> pd.Series:
    """
    Long-only mining valuation tilt (UN-SHIFTED), continuous exposure in [0,1].

    Steps:
      1) edge = log(price/cost)  (or simple ratio edge)
      2) smooth edge with EMA (slow)
      3) rolling z-score of smoothed edge
      4) map z into exposure:
           z <= z_lo  -> max_exposure
           z >= z_hi  -> min_exposure
           linear in between

    Why this helps:
      - avoids binary whipsaws
      - keeps exposure near 1 most of the time (tracks buyhold)
      - still de-risks when price is very rich vs mining cost
    """
    idx = df.index
    price = df[price_column].astype(float).replace([np.inf, -np.inf], np.nan)
    cost = df[cost_column].astype(float).replace([np.inf, -np.inf], np.nan)
    cost = cost.where(cost > 0)

    if use_log_edge:
        edge = np.log(price / cost)
    else:
        edge = (price - cost) / cost

    edge = edge.replace([np.inf, -np.inf], np.nan)

    # smooth to reduce churn
    span = int(max(5, smooth_span))
    edge_s = edge.ewm(span=span, min_periods=max(10, span // 3), adjust=False).mean()

    z = _rolling_zscore(edge_s, int(z_window))

    # map z -> exposure (invert: expensive => lower exposure)
    z_lo = float(z_lo)
    z_hi = float(z_hi)
    if z_hi <= z_lo:
        raise ValueError("z_hi must be > z_lo")

    min_exposure = float(min_exposure)
    max_exposure = float(max_exposure)
    min_exposure = float(np.clip(min_exposure, 0.0, 1.0))
    max_exposure = float(np.clip(max_exposure, 0.0, 1.0))

    # linear interpolation: cheap (low z) => high exposure
    # exposure = max_exposure at z_lo, min_exposure at z_hi
    expo = pd.Series(np.nan, index=idx, dtype=float)
    t = (z - z_lo) / (z_hi - z_lo)  # 0 at z_lo, 1 at z_hi
    expo = max_exposure + (min_exposure - max_exposure) * t
    expo = expo.clip(lower=min_exposure, upper=max_exposure)

    # optional monthly trend gate (cap exposure in risk-off)
    if use_trend_gate:
        risk_on = _risk_on_gate_from_monthly_ma(price.dropna(), ma_months=gate_ma_months).reindex(idx).fillna(0.0)
        cap = float(np.clip(risk_off_cap, 0.0, 1.0))
        expo = expo.where(risk_on > 0.5, other=np.minimum(expo, cap))

    return expo.fillna(min_exposure).clip(0.0, 1.0)