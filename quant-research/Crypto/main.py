import pandas as pd

from config import *
from Data.raw_data_loader import load_raw_crypto_csv, load_ibit_with_mining_cost

from Models.amma import amma_signal
from Models.zscore import zscore_signal
from Models.trend import trend_signal
from Models.ou import ou_signal
from Models.mining import mining_signal

from Backtest.engine import run_backtest
from Backtest.metrics import build_summary_table


# =========================================================
# Select Model
# =========================================================

ASSET_TO_TRADE = "IBIT"     # "IBIT" or "BTC"
MODEL_TO_RUN = "trend"      # "amma", "zscore", "trend", "ou", "mining", "buyhold"


# =========================================================
# Load Data
# =========================================================

if ASSET_TO_TRADE.upper() == "IBIT":
    df = load_ibit_with_mining_cost(
        ibit_path=IBIT_PATH,
        cleaned_crypto_path=DATA_PATH,
        forward_fill_mining_cost=True,
    ).rename(
        columns={
            "close": PRICE_COLUMN_IBIT,
            "mining_cost": COST_COLUMN_MINE,
        }
    )
    price_col = PRICE_COLUMN_IBIT
else:
    df = load_raw_crypto_csv(DATA_PATH, start_date=DATA_START_DATE)
    price_col = PRICE_COLUMN_BTC


# =========================================================
# Signal Wrapper (long-only, UN-SHIFTED)
# Engine shifts + clips.
# =========================================================

def signal_wrapper(full_df: pd.DataFrame) -> pd.Series:
    price = full_df[price_col].astype(float)

    if MODEL_TO_RUN == "buyhold":
        return pd.Series(1.0, index=full_df.index)

    if MODEL_TO_RUN == "amma":
        return amma_signal(
            full_df,
            price_column=price_col,
            momentum_weights={20: 0.25, 60: 0.25, 120: 0.25, 252: 0.25},
            threshold=0.0,
            normalize_weights=True,
        )

    if MODEL_TO_RUN == "zscore":
        vol_target = VOL_TARGET if USE_VOL_TARGET else None
        return zscore_signal(
            price_series=price,
            window=ZSCORE_WINDOW,
            entry_z=ZSCORE_ENTRY_Z,
            exit_z=ZSCORE_EXIT_Z,
            long_short=False,
            max_leverage=1.0,
            vol_window=VOL_WINDOW,
            vol_target=vol_target,
        )

    if MODEL_TO_RUN == "trend":
        vol_target = VOL_TARGET if USE_VOL_TARGET else None
        return trend_signal(
            full_df,
            price_column=price_col,
            fast_window=TREND_FAST_WINDOW,
            slow_window=TREND_SLOW_WINDOW,
            long_only=True,
            leverage_aggressive=TREND_AGGRESSIVE,
            leverage_neutral=TREND_NEUTRAL,
            leverage_defensive=TREND_DEFENSIVE,
            vol_window=VOL_WINDOW,
            vol_target=vol_target,
            max_leverage=1.0,
        )

    if MODEL_TO_RUN == "ou":
        return ou_signal(
            price_series=price,
            window=OU_WINDOW,
            entry_z=OU_ENTRY_Z,
            exit_z=OU_EXIT_Z,
            long_short=False,
        )

    if MODEL_TO_RUN == "mining":
        if ASSET_TO_TRADE.upper() != "IBIT":
            raise ValueError("Mining model expects IBIT + mining cost aligned.")
        return mining_signal(
            full_df,
            price_column=price_col,
            cost_column=COST_COLUMN_MINE,
            z_window=MINING_Z_WINDOW,
            entry_z=MINING_ENTRY_Z,
            exit_z=MINING_EXIT_Z,
            use_log_edge=MINING_USE_LOG_EDGE,
        )

    raise ValueError(f"Invalid MODEL_TO_RUN: {MODEL_TO_RUN}")


# =========================================================
# In-Sample Backtests (Compare vs Buy&Hold and AMMA)
# =========================================================

pos_buyhold = pd.Series(1.0, index=df.index)

pos_amma = amma_signal(
    df,
    price_column=price_col,
    momentum_weights={20: 0.25, 60: 0.25, 120: 0.25, 252: 0.25},
    threshold=0.0,
    normalize_weights=True,
)

pos_model = signal_wrapper(df)

res_buyhold = run_backtest(
    df=df,
    price_col=price_col,
    position=pos_buyhold,
    fee_bps=FEE_BPS,
    slippage_bps=SLIPPAGE_BPS,
    annual_borrow_rate=ANNUAL_BORROW_RATE,
    long_only=True,
    leverage_cap=1.0,
)

res_amma = run_backtest(
    df=df,
    price_col=price_col,
    position=pos_amma,
    fee_bps=FEE_BPS,
    slippage_bps=SLIPPAGE_BPS,
    annual_borrow_rate=ANNUAL_BORROW_RATE,
    long_only=True,
    leverage_cap=1.0,
)

res_model = run_backtest(
    df=df,
    price_col=price_col,
    position=pos_model,
    fee_bps=FEE_BPS,
    slippage_bps=SLIPPAGE_BPS,
    annual_borrow_rate=ANNUAL_BORROW_RATE,
    long_only=True,
    leverage_cap=1.0,
)

summary_table = build_summary_table({
    "Buy&Hold Net": {"returns": res_buyhold["net_returns"], "position": res_buyhold["position"]},
    "AMMA Net":     {"returns": res_amma["net_returns"],    "position": res_amma["position"]},
    f"{MODEL_TO_RUN} Net": {"returns": res_model["net_returns"], "position": res_model["position"]},
})

print("==== In-Sample Results (Net) ====")
print(summary_table)