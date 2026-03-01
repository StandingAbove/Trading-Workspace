# Models/amma.py
from __future__ import annotations

from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
    from polars import LazyFrame
except ModuleNotFoundError:  # optional
    pl = None
    LazyFrame = object

try:
    from common.bundles import ModelStateBundle
except ModuleNotFoundError:
    ModelStateBundle = object


# =========================================================
# 1) Project-facing AMMA (pandas) — long-only, unshifted
# =========================================================

def _normalize_positive_weights(momentum_weights: Dict[int, float]) -> Dict[int, float]:
    if not momentum_weights:
        raise ValueError("momentum_weights must contain at least one window.")

    w = {int(k): float(v) for k, v in momentum_weights.items()}
    if any(v < 0 for v in w.values()):
        raise ValueError("For long-only AMMA, all momentum weights must be >= 0.")

    s = sum(w.values())
    if s <= 0:
        raise ValueError("Sum of momentum weights must be > 0.")

    return {k: v / s for k, v in w.items()}


def amma_signal(
    data: Union[pd.Series, pd.DataFrame],
    price_column: str = "IBIT_close",
    momentum_weights: Dict[int, float] = None,
    threshold: float = 0.0,
    normalize_weights: bool = True,
    update_freq: str = "daily",  # "daily" or "monthly"
) -> pd.Series:
    """
    Long-only AMMA position in [0, 1], UN-SHIFTED.

    For each lookback window 'w':
      mom_w = pct_change(w)
      component_w = weight_w if mom_w > threshold else 0

    position = sum(component_w)
    If normalize_weights=True, weights are normalized to sum to 1, so position is in [0,1].

    update_freq="monthly":
      - Compute momentum on month-end closes
      - Forward-fill within each month (low turnover)

    IMPORTANT:
      - This function does NOT shift.
      - Your engine enforces execution lag via shift(1) and clips to [0,1].
    """
    if momentum_weights is None:
        # safe default: sums to 1
        momentum_weights = {20: 0.25, 60: 0.25, 120: 0.25, 252: 0.25}

    weights = _normalize_positive_weights(momentum_weights) if normalize_weights else {
        int(k): float(v) for k, v in momentum_weights.items()
    }

    if isinstance(data, pd.Series):
        price = data.astype(float).copy()
        idx = price.index
    elif isinstance(data, pd.DataFrame):
        price = data[price_column].astype(float).copy()
        idx = data.index
    else:
        raise TypeError("data must be a pandas Series or DataFrame")

    price = price.replace([np.inf, -np.inf], np.nan)

    if update_freq == "monthly":
        # month-end close (pandas version safe)
        try:
            monthly = price.resample("ME").last()
        except ValueError:
            monthly = price.resample("M").last()

        pos_m = pd.Series(0.0, index=monthly.index, dtype=float)

        for window, weight in weights.items():
            # convert trading-day window to months (20->1, 60->3, 120->6, 252->12)
            months = max(1, int(round(int(window) / 21)))
            mom = monthly.pct_change(months)
            pos_m += (mom > float(threshold)).astype(float) * float(weight)

        pos_d = pos_m.reindex(price.index, method="ffill").fillna(0.0)
        return pos_d.reindex(idx).fillna(0.0).clip(0.0, 1.0)

    if update_freq != "daily":
        raise ValueError("update_freq must be 'daily' or 'monthly'")

    # daily behavior (existing)
    pos = pd.Series(0.0, index=idx, dtype=float)
    for window, weight in weights.items():
        mom = price.pct_change(int(window))
        pos += (mom > float(threshold)).astype(float) * float(weight)

    return pos.fillna(0.0).clip(0.0, 1.0)


# =========================================================
# 2) Existing Polars AMMA (kept for compatibility)
# =========================================================

def AMMA(
    ticker: str,
    momentum_weights: Dict[int, float],
    threshold: float = 0.0,
    long_enabled: bool = True,
    short_enabled: bool = False,
) -> Callable[[Any], Any]:
    """
    Original model-state AMMA (Polars).
    Kept as-is for your other pipeline. Not used by main.py backtests.
    """

    def run_model(bundle: ModelStateBundle) -> LazyFrame:
        if pl is None:
            raise ImportError("polars is required for AMMA() model-state execution.")

        if not momentum_weights:
            raise ValueError("momentum_weights must contain at least one lookback window.")

        lf = bundle.model_state.lazy()
        sig_frames = []

        for window, weight in momentum_weights.items():
            colname = f"close_momentum_{window}"

            sig = (
                lf.filter(pl.col("ticker") == ticker)
                .select([pl.col("date"), pl.col(colname).alias("sig")])
            )

            long_cond = (pl.col("sig") > threshold) if long_enabled else None
            short_cond = (pl.col("sig") < -threshold) if short_enabled else None

            expr = pl.lit(0.0)
            if long_enabled and short_enabled:
                expr = (
                    pl.when(long_cond.fill_null(False)).then(pl.lit(1.0) * weight)
                    .when(short_cond.fill_null(False)).then(pl.lit(-1.0) * weight)
                    .otherwise(pl.lit(0.0))
                )
            elif long_enabled:
                expr = pl.when(long_cond.fill_null(False)).then(pl.lit(1.0) * weight).otherwise(pl.lit(0.0))
            elif short_enabled:
                expr = pl.when(short_cond.fill_null(False)).then(pl.lit(-1.0) * weight).otherwise(pl.lit(0.0))

            weighted_sig = sig.select([pl.col("date"), expr.cast(pl.Float64).alias(f"sig_{window}")])
            sig_frames.append(weighted_sig)

        combined = sig_frames[0]
        for frame in sig_frames[1:]:
            combined = combined.join(frame, on="date", how="inner")

        weight_cols = [f"sig_{w}" for w in momentum_weights.keys()]
        return combined.with_columns(sum([pl.col(c) for c in weight_cols]).alias(ticker)).select(["date", ticker])

    return run_model


# =========================================================
# 3) CSV helper (updated to be UN-SHIFTED)
# =========================================================

def amma_from_ibit_csv(
    ibit_csv_path: str,
    momentum_weights: Dict[int, float],
    threshold: float = 0.0,
    normalize_weights: bool = True,
    update_freq: str = "daily",
) -> pd.DataFrame:
    """
    Build AMMA signal directly from the IBIT CSV schema.

    Output columns: Date, price, amma_weight (UN-SHIFTED in [0,1]).
    (Engine should shift positions, not this function.)
    """
    ibit = pd.read_csv(ibit_csv_path).copy()

    if "Date" not in ibit.columns or "Price" not in ibit.columns:
        raise ValueError("IBIT CSV must contain 'Date' and 'Price' columns.")

    ibit["Date"] = pd.to_datetime(ibit["Date"], format="%m/%d/%y", errors="coerce")
    ibit["price"] = (
        ibit["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .astype(float)
    )
    ibit = ibit.dropna(subset=["Date", "price"]).sort_values("Date").reset_index(drop=True)

    s = amma_signal(
        data=ibit["price"],
        momentum_weights=momentum_weights,
        threshold=threshold,
        normalize_weights=normalize_weights,
        update_freq=update_freq,
    )

    ibit["amma_weight"] = s.values
    return ibit[["Date", "price", "amma_weight"]]