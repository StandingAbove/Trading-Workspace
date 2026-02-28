import numpy as np
import pandas as pd


def trend_signal(
    df: pd.DataFrame,
    price_column: str = "IBIT_close",
    mode: str = "faber_10m",          # "faber_10m" or "tsmom_12m"
    ma_months: int = 10,             # for faber_10m
    mom_months: int = 12,            # for tsmom_12m
    exposure_on: float = 1.0,        # long-only exposure when risk-on
    exposure_off: float = 0.0,       # long-only exposure when risk-off
) -> pd.Series:
    """
    Long-only monthly trend filter (UN-SHIFTED).
    Engine should shift(1) and clip to [0,1].

    mode="faber_10m":
      - monthly close > 10-month SMA => risk-on
      - else risk-off
      (classic timing model)  :contentReference[oaicite:2]{index=2}

    mode="tsmom_12m":
      - past 12-month return > 0 => risk-on
      - else risk-off
      (time-series momentum) :contentReference[oaicite:3]{index=3}
    """
    price = df[price_column].astype(float).replace([np.inf, -np.inf], np.nan)
    price = price.dropna()

    # Monthly close series
    monthly_close = price.resample("ME").last()
    if mode == "faber_10m":
        ma = monthly_close.rolling(int(ma_months), min_periods=int(ma_months)).mean()
        sig_m = (monthly_close > ma).astype(float)

    elif mode == "tsmom_12m":
        mom = monthly_close.pct_change(int(mom_months))
        sig_m = (mom > 0.0).astype(float)

    else:
        raise ValueError("mode must be 'faber_10m' or 'tsmom_12m'")

    # Map monthly signal to daily, forward-fill within month
    sig_d = sig_m.reindex(price.index, method="ffill").fillna(0.0)

    # Long-only exposures, no leverage
    on = float(min(max(exposure_on, 0.0), 1.0))
    off = float(min(max(exposure_off, 0.0), 1.0))

    pos = sig_d * on + (1.0 - sig_d) * off
    pos = pos.reindex(df.index).fillna(0.0).clip(0.0, 1.0)

    return pos