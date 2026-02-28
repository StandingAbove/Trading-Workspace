import numpy as np
import pandas as pd

from Backtest.costs import apply_costs, annual_turnover

DAYS_PER_YEAR = 365


def compute_strategy_returns(prices: pd.Series, position: pd.Series) -> pd.Series:
    """
    Gross strategy returns using close-to-close returns.
    Assumes 'position' is aligned to prices and already execution-lagged.
    """
    prices = prices.astype(float)
    ret = prices.pct_change().fillna(0.0)

    position = position.reindex(prices.index).fillna(0.0).astype(float)
    return position * ret


def run_backtest(
    df: pd.DataFrame,
    price_col: str,
    position: pd.Series,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    annual_borrow_rate: float = 0.0,
    long_only: bool = True,
    leverage_cap: float = 1.0,
) -> dict:
    """
    Backtest with engine-enforced:
      - execution lag (shift 1)
      - no leverage (clip to [0, 1] for long-only)
    """
    prices = df[price_col].astype(float).copy()

    # Align + fill
    raw_pos = position.reindex(prices.index).fillna(0.0).astype(float)

    # Execution lag (prevents lookahead)
    exec_pos = raw_pos.shift(1).fillna(0.0)

    # Enforce leverage rules
    leverage_cap = float(leverage_cap)
    if long_only:
        exec_pos = exec_pos.clip(0.0, leverage_cap)
    else:
        exec_pos = exec_pos.clip(-leverage_cap, leverage_cap)

    gross_returns = compute_strategy_returns(prices, exec_pos)

    net_returns = apply_costs(
        returns=gross_returns,
        position=exec_pos,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        annual_borrow_rate=annual_borrow_rate,
    )

    gross_equity = (1.0 + gross_returns).cumprod()
    net_equity = (1.0 + net_returns).cumprod()

    return {
        "position": exec_pos,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "gross_equity": gross_equity,
        "net_equity": net_equity,
        "annual_turnover": float(annual_turnover(exec_pos)),
    }