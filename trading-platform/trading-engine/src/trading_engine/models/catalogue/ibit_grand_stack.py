import numpy as np
import pandas as pd

from .ibit_overlays import compute_overlay
from . import amma, trend, ou, zscore

try:
    from . import mining
    HAS_MINING = True
except Exception:
    HAS_MINING = False


W_AMMA = 0.55
W_TREND = 0.20
W_OU = 0.10
W_Z = 0.10
W_MINING = 0.05


def generate_positions(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if "IBIT_close" not in df.columns:
        raise KeyError("IBIT_close missing")

    px = df["IBIT_close"].astype(float)

    overlay = compute_overlay(px)

    # positions from each model (each should already be shifted)
    p_amma = amma.position(df) if hasattr(amma, "position") else amma.pos_amma(df)
    p_trend = trend.position(df) if hasattr(trend, "position") else trend.pos_trend(df)
    p_ou = ou.position(df) if hasattr(ou, "position") else ou.pos_ou(df)
    p_z = zscore.position(df) if hasattr(zscore, "position") else zscore.pos_zscore(df)

    if HAS_MINING:
        p_m = mining.position(df) if hasattr(mining, "position") else mining.pos_mining(df)
    else:
        p_m = pd.Series(0.0, index=df.index)

    # apply overlay ONLY to AMMA leg
    p_amma_overlay = (p_amma * overlay).clip(0.0, 1.0)

    # grand linear combo
    pos = (
        W_AMMA * p_amma_overlay
        + W_TREND * p_trend
        + W_OU * p_ou
        + W_Z * p_z
        + W_MINING * p_m
    )

    pos = pos.clip(0.0, 1.0)
    pos.name = "IBIT-US"

    out = pd.DataFrame({"date": df.index, "IBIT-US": pos})
    return out
