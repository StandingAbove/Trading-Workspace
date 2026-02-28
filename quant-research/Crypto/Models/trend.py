import numpy as np
import pandas as pd


def trend_signal(
    df: pd.DataFrame,
    price_column: str = "IBIT_close",
    long_only: bool = True,  # kept for compatibility (we enforce long-only)
    leverage_aggressive: float = 1.0,  # capped to <= 1.0
    leverage_neutral: float = 1.0,     # used as risk-on exposure (<= 1.0)
    leverage_defensive: float = 0.0,   # used as risk-off exposure (<= 1.0)
    max_leverage: float = 1.0,
    # --- core logic knobs ---
    ema_window: int = 200,     # slow regime filter
    band: float = 0.01,        # 1% band to reduce whipsaw
    dd_window: int = 90,       # rolling peak window for drawdown
    dd_stop: float = 0.18,     # go risk-off if drawdown worse than -18%
    mom_short: int = 63,       # ~3 months momentum
    mom_long: int = 252,       # ~1 year momentum
    # --- optional: scale DOWN in high vol (never up) ---
    vol_window: int = 30,
    vol_target: float | None = None,
    # --- optional: minimum hold to reduce flip-flops ---
    min_hold_days: int = 5,
) -> pd.Series:
    """
    Crash-filtered long-only trend.

    Idea:
      - Default: stay invested (risk-on) when regime is healthy.
      - Risk-off only when a real crash shows up:
          (drawdown < -dd_stop) AND (price below EMA with a band)
        or when long momentum turns negative while price is below EMA.

    Output:
      UN-SHIFTED target position in [0, 1].
      Your engine shifts(1) and clips, enforcing no lookahead/no leverage.
    """
    price = df[price_column].astype(float).replace([np.inf, -np.inf], np.nan)
    idx = df.index

    # Smooth regime proxy
    ema = price.ewm(span=int(ema_window), adjust=False, min_periods=max(10, ema_window // 5)).mean()

    # Drawdown from rolling peak
    peak = price.rolling(int(dd_window), min_periods=max(10, dd_window // 3)).max()
    dd = (price / peak) - 1.0
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Momentum filters
    mom_s = price.pct_change(int(mom_short)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mom_l = price.pct_change(int(mom_long)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Regime: price above EMA (with hysteresis band)
    above = price > (ema * (1.0 + float(band)))
    below = price < (ema * (1.0 - float(band)))

    # Risk-off conditions (crash / bear)
    crash = (dd < -float(dd_stop)) & below
    bear = (mom_l < 0.0) & below

    # Risk-on condition (re-enter)
    risk_on = above & (mom_s > 0.0)

    # State machine with min-hold
    state = pd.Series(0.0, index=idx, dtype=float)
    in_pos = 0.0
    hold = 0

    for i in range(len(state)):
        if hold > 0:
            hold -= 1
            state.iloc[i] = in_pos
            continue

        if in_pos == 0.0:
            if bool(risk_on.iloc[i]):
                in_pos = 1.0
                hold = int(min_hold_days)
        else:
            if bool(crash.iloc[i]) or bool(bear.iloc[i]):
                in_pos = 0.0
                hold = int(min_hold_days)

        state.iloc[i] = in_pos

    # Exposures (long-only, no leverage)
    on_expo = float(min(leverage_neutral, 1.0))
    off_expo = float(min(leverage_defensive, 1.0))
    pos = state * on_expo + (1.0 - state) * off_expo

    # Optional vol scale-down only (never lever up)
    if vol_target is not None and np.isfinite(vol_target) and float(vol_target) > 0 and int(vol_window) > 1:
        ret = price.pct_change()
        vol = ret.rolling(int(vol_window), min_periods=max(5, vol_window // 3)).std()
        scale = (float(vol_target) / vol).replace([np.inf, -np.inf], np.nan)
        scale = scale.clip(0.0, 1.0)  # never > 1
        pos = (pos * scale).fillna(0.0)

    # Final caps
    pos = pos.fillna(0.0)
    pos = pos.clip(0.0, float(min(max_leverage, 1.0)))
    return pos