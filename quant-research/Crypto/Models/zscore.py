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


def _monthly_risk_on_gate(price: pd.Series, ma_months: int = 10) -> pd.Series:
    ma_months = int(ma_months)

    try:
        monthly_close = price.resample("ME").last()
    except ValueError:
        monthly_close = price.resample("M").last()

    monthly_ma = monthly_close.rolling(ma_months, min_periods=ma_months).mean()
    risk_on_monthly = (monthly_close > monthly_ma).fillna(False)
    risk_on_daily = risk_on_monthly.reindex(price.index).ffill()
    risk_on_daily = risk_on_daily.where(risk_on_daily.notna(), False).astype(bool)
    return risk_on_daily


def zscore_signal(
    price_series: pd.Series,
    window: int = 180,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    long_short: bool = False,     # kept for compatibility; we enforce long-only anyway
    max_leverage: float = 1.0,    # hard-capped to <= 1.0
    vol_window: int = 30,
    vol_target: float = 0.02,
    min_hold_days: int = 5,
    cooldown_days: int = 3,
    allow_in_risk_on: bool = False,
    sideways_band: float | None = None,
    regime_ma_months: int = 10,
) -> pd.Series:
    """
    Long-only mean reversion on rolling z-score of price.

    Logic:
      z = (price - mean) / std
      - Enter long when z < -entry_z
      - Exit when z > -exit_z

    Output is UN-SHIFTED target position in [0, 1].
    """
    price = price_series.astype(float).replace([np.inf, -np.inf], np.nan)
    logp = np.log(price.replace(0.0, np.nan))

    mu = _rolling_mean(logp, window)
    sd = _rolling_std(logp, window)
    sd = sd.where(sd >= 1e-8, np.nan)

    z = (logp - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan)

    risk_on = _monthly_risk_on_gate(price, ma_months=regime_ma_months)
    allow_risk_on = bool(allow_in_risk_on)

    if sideways_band is not None and np.isfinite(sideways_band) and float(sideways_band) > 0:
        try:
            monthly_close = price.resample("ME").last()
        except ValueError:
            monthly_close = price.resample("M").last()
        monthly_ma = monthly_close.rolling(int(regime_ma_months), min_periods=int(regime_ma_months)).mean()
        ma_daily = monthly_ma.reindex(price.index, method="ffill")
        distance = ((price / ma_daily) - 1.0).abs()
        sideways = (distance <= float(sideways_band)).fillna(False)
        trade_gate = (~risk_on) | sideways
    else:
        trade_gate = (~risk_on)

    pos = pd.Series(0.0, index=price.index)
    in_pos = 0.0
    hold_days = 0
    cooldown_left = 0

    min_hold_days = max(0, int(min_hold_days))
    cooldown_days = max(0, int(cooldown_days))

    for i in range(len(z)):
        zi = z.iloc[i]
        can_trade = bool(trade_gate.iloc[i]) or allow_risk_on

        if not can_trade:
            in_pos = 0.0
            hold_days = 0
            if cooldown_days > 0:
                cooldown_left = max(cooldown_left, cooldown_days)
            pos.iloc[i] = 0.0
            if cooldown_left > 0:
                cooldown_left -= 1
            continue

        if np.isnan(zi):
            pos.iloc[i] = in_pos
            if in_pos > 0.0:
                hold_days += 1
            if cooldown_left > 0:
                cooldown_left -= 1
            continue

        if cooldown_left <= 0:
            if in_pos == 0.0:
                if zi < -float(entry_z):
                    in_pos = 1.0
                    hold_days = 0
                    cooldown_left = cooldown_days
            else:
                if hold_days >= min_hold_days and zi > -float(exit_z):
                    in_pos = 0.0
                    hold_days = 0
                    cooldown_left = cooldown_days

        if in_pos > 0.0:
            hold_days += 1

        pos.iloc[i] = in_pos

        if cooldown_left > 0:
            cooldown_left -= 1

    # Optional: scale DOWN in high vol (never scale up)
    vol_window = int(vol_window)
    if vol_window > 1 and vol_target is not None and np.isfinite(vol_target) and vol_target > 0:
        ret = price.pct_change()
        vol = ret.rolling(vol_window, min_periods=max(3, vol_window // 3)).std()
        scale = (vol_target / vol).replace([np.inf, -np.inf], np.nan)
        scale = scale.clip(0.0, 1.0)  # never lever up
        pos = (pos * scale).fillna(0.0)

    pos = pos.fillna(0.0)
    pos = pos.clip(0.0, float(min(max_leverage, 1.0)))
    return pos
