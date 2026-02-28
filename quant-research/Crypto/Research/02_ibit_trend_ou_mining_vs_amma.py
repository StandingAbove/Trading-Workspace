"""Research 02: IBIT equity-curve comparison.

Build and compare:
1) Buy & Hold IBIT
2) AMMA (using same implementation as trading-engine model logic)
3) Trend + OU + Mining ensemble on IBIT with mining cost from cleaned crypto data

Outputs:
- prints a summary table
- writes equity curve plot to `Crypto/Research/artifacts/02_ibit_equity_curve.png`
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Backtest.amma import amma_from_ibit_csv
from Data.raw_data_loader import load_ibit_with_mining_cost
from Models.mining import mining_signal
from Models.ou import ou_signal
from Models.trend import trend_signal

def perf_summary(returns: pd.Series) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

    equity = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(252.0)
    sharpe = (r.mean() * 252.0) / vol if vol > 0 else np.nan
    dd = (equity / equity.cummax() - 1.0).min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(dd)}


def main() -> None:
    root = ROOT
    ibit_path = root / "Data" / "IBIT_Data.csv"
    cleaned_path = root / "Data" / "cleaned_crypto_data.csv"

    # AMMA directly from IBIT csv (matches AMMA momentum-aggregation logic)
    amma = amma_from_ibit_csv(
        str(ibit_path),
        momentum_weights={5: 0.25, 10: 0.25, 20: 0.25, 60: 0.25},
        threshold=0.0,
        long_enabled=True,
        short_enabled=False,
    ).rename(columns={"Date": "date"})

    # IBIT + mining-cost panel
    ibit = load_ibit_with_mining_cost(str(ibit_path), str(cleaned_path), forward_fill_mining_cost=True)

    df = ibit.copy()
    df["close"] = df["close"].astype(float)

    # Build temporary schema for existing model functions
    model_df = pd.DataFrame(index=df.index)
    model_df["BTC-USD_close"] = df["close"]
    model_df["COST_TO_MINE"] = df["mining_cost"]

    trend_pos = trend_signal(
        model_df,
        price_column="BTC-USD_close",
        fast_window=20,
        slow_window=100,
        long_only=True,
        leverage_aggressive=1.0,
        leverage_neutral=0.5,
        leverage_defensive=0.0,
    ).reindex(df.index).fillna(0.0)

    ou_pos = ou_signal(
        model_df["BTC-USD_close"],
        window=120,
        entry_z=1.5,
        exit_z=0.3,
        long_short=False,
    ).reindex(df.index).fillna(0.0)

    mining_pos = mining_signal(
        model_df,
        z_window=180,
        entry_z=0.8,
        exit_z=0.0,
        use_log_edge=True,
    ).reindex(df.index).fillna(0.0)

    combo_position = ((trend_pos + ou_pos + mining_pos) / 3.0).clip(0.0, 1.0)

    daily_ret = df["close"].pct_change().fillna(0.0)

    # AMMA alignment on IBIT date index
    amma = amma.set_index("date").reindex(df.index).fillna(0.0)

    ret_bh = daily_ret
    ret_amma = amma["amma_position"].astype(float) * daily_ret
    ret_combo = combo_position * daily_ret

    results = pd.DataFrame(
        {
            "buy_hold": ret_bh,
            "amma": ret_amma,
            "trend_ou_mining": ret_combo,
        },
        index=df.index,
    )

    equity = (1.0 + results).cumprod()

    summary = pd.DataFrame(
        {
            "Buy & Hold": perf_summary(results["buy_hold"]),
            "AMMA": perf_summary(results["amma"]),
            "Trend+OU+Mining": perf_summary(results["trend_ou_mining"]),
        }
    )
    print("=== IBIT Comparison Summary ===")
    print(summary)

    out_dir = root / "Research" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "02_ibit_equity_curve.png"

    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity["buy_hold"], label="Buy & Hold", linewidth=2)
    plt.plot(equity.index, equity["amma"], label="AMMA", linewidth=2)
    plt.plot(equity.index, equity["trend_ou_mining"], label="Trend + OU + Mining", linewidth=2)
    plt.title("IBIT Equity Curves: Buy & Hold vs AMMA vs Trend+OU+Mining")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"Saved equity curve to: {out_path}")


if __name__ == "__main__":
    main()
