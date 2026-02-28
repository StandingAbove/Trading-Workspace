"""Research 02: realistic IBIT strategy comparison (no leverage).

Compares IBIT equity curves for:
- Buy & Hold
- AMMA
- Trend + OU + Mining
- OU + ZScore
- Stochastic Tactical
- Grand All (average of all model positions)

Design choices to keep results realistic:
- no leverage (all positions clipped to [0, 1])
- one-bar execution lag (model functions already lag; we keep alignment strict)
- transaction costs applied on position changes
- mining cost sourced from cleaned_crypto_data.csv
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
from Models.zscore import zscore_signal


def _stochastic_tactical_position(price: pd.Series, k_window: int = 14, d_window: int = 3) -> pd.Series:
    """Long-only stochastic tactical position with lag and hysteresis."""
    low = price.rolling(k_window, min_periods=k_window).min()
    high = price.rolling(k_window, min_periods=k_window).max()
    denom = (high - low).replace(0.0, np.nan)
    k = 100.0 * ((price - low) / denom)
    d = k.rolling(d_window, min_periods=d_window).mean()

    pos = pd.Series(0.0, index=price.index)
    current = 0.0
    for i in range(len(price)):
        ki = k.iloc[i]
        di = d.iloc[i]
        if np.isnan(ki) or np.isnan(di):
            pos.iloc[i] = current
            continue

        # Enter on momentum confirmation, exit on deterioration
        if current == 0.0 and (ki > 55.0 and di > 50.0):
            current = 1.0
        elif current == 1.0 and (ki < 45.0 and di < 50.0):
            current = 0.0
        pos.iloc[i] = current

    return pos.shift(1).fillna(0.0).clip(0.0, 1.0)


def _apply_costs(returns: pd.Series, position: pd.Series, fee_bps: float = 5.0, slippage_bps: float = 2.0) -> pd.Series:
    """Simple transaction-cost model: charged on absolute daily position change."""
    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    costs = turnover * cost_rate
    return (returns - costs).fillna(0.0)


def _perf_summary(returns: pd.Series) -> dict:
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
    ibit_path = ROOT / "Data" / "IBIT_Data.csv"
    cleaned_path = ROOT / "Data" / "cleaned_crypto_data.csv"

    # Load IBIT + mining cost
    panel = load_ibit_with_mining_cost(str(ibit_path), str(cleaned_path), forward_fill_mining_cost=True)
    panel = panel.sort_index().copy()
    panel["close"] = panel["close"].astype(float)

    model_df = pd.DataFrame(index=panel.index)
    model_df["BTC-USD_close"] = panel["close"]
    model_df["COST_TO_MINE"] = panel["mining_cost"]

    # AMMA base (weights sum to 1.0 => no leverage)
    amma = amma_from_ibit_csv(
        str(ibit_path),
        momentum_weights={5: 0.25, 10: 0.25, 20: 0.25, 60: 0.25},
        threshold=0.0,
        long_enabled=True,
        short_enabled=False,
    ).rename(columns={"Date": "date"}).set_index("date")
    amma_pos = amma.reindex(panel.index)["amma_position"].fillna(0.0).clip(0.0, 1.0)

    trend_pos = trend_signal(
        model_df,
        price_column="BTC-USD_close",
        fast_window=20,
        slow_window=100,
        long_only=True,
        leverage_aggressive=1.0,
        leverage_neutral=0.5,
        leverage_defensive=0.0,
    ).reindex(panel.index).fillna(0.0).clip(0.0, 1.0)

    ou_pos = ou_signal(
        model_df["BTC-USD_close"],
        window=120,
        entry_z=1.5,
        exit_z=0.4,
        long_short=False,
    ).reindex(panel.index).fillna(0.0).clip(0.0, 1.0)

    mining_pos = mining_signal(
        model_df,
        z_window=180,
        entry_z=0.8,
        exit_z=0.1,
        use_log_edge=True,
    ).reindex(panel.index).fillna(0.0).clip(0.0, 1.0)

    z_pos = zscore_signal(
        model_df["BTC-USD_close"],
        resid_window=120,
        entry_z=1.25,
        exit_z=0.4,
        long_short=False,
        use_vol_target=False,
        max_leverage=1.0,
    ).reindex(panel.index).fillna(0.0).clip(0.0, 1.0)

    stoch_pos = _stochastic_tactical_position(panel["close"], k_window=14, d_window=3)

    # Requested combo
    trend_ou_mining_pos = ((trend_pos + ou_pos + mining_pos) / 3.0).clip(0.0, 1.0)

    # Additional realism checks requested
    ou_z_pos = ((ou_pos + z_pos) / 2.0).clip(0.0, 1.0)
    grand_all_pos = ((amma_pos + trend_pos + ou_pos + mining_pos + z_pos + stoch_pos) / 6.0).clip(0.0, 1.0)

    ret = panel["close"].pct_change().fillna(0.0)

    raw = pd.DataFrame({
        "buy_hold": ret,
        "amma": amma_pos * ret,
        "trend_ou_mining": trend_ou_mining_pos * ret,
        "ou_zscore": ou_z_pos * ret,
        "stochastic_tactical": stoch_pos * ret,
        "grand_all": grand_all_pos * ret,
    }, index=panel.index)

    results = pd.DataFrame(index=raw.index)
    results["buy_hold"] = raw["buy_hold"]
    for c in ["amma", "trend_ou_mining", "ou_zscore", "stochastic_tactical", "grand_all"]:
        pos = {
            "amma": amma_pos,
            "trend_ou_mining": trend_ou_mining_pos,
            "ou_zscore": ou_z_pos,
            "stochastic_tactical": stoch_pos,
            "grand_all": grand_all_pos,
        }[c]
        results[c] = _apply_costs(raw[c], pos, fee_bps=5.0, slippage_bps=2.0)

    equity = (1.0 + results).cumprod()

    summary = pd.DataFrame({
        "Buy & Hold": _perf_summary(results["buy_hold"]),
        "AMMA": _perf_summary(results["amma"]),
        "Trend+OU+Mining": _perf_summary(results["trend_ou_mining"]),
        "OU+ZScore": _perf_summary(results["ou_zscore"]),
        "Stochastic Tactical": _perf_summary(results["stochastic_tactical"]),
        "Grand All": _perf_summary(results["grand_all"]),
    })

    print("=== IBIT Realistic Comparison Summary (No Leverage, Cost-Aware) ===")
    print(summary)

    out_dir = ROOT / "Research" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: requested view
    out1 = out_dir / "02_ibit_equity_curve_core.png"
    plt.figure(figsize=(11, 6))
    plt.plot(equity.index, equity["buy_hold"], label="Buy & Hold", linewidth=2)
    plt.plot(equity.index, equity["amma"], label="AMMA", linewidth=2)
    plt.plot(equity.index, equity["trend_ou_mining"], label="Trend + OU + Mining", linewidth=2)
    plt.title("IBIT Equity Curves (No Leverage): Buy & Hold vs AMMA vs Trend+OU+Mining")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    plt.close()

    # Plot 2: extended realism checks requested
    out2 = out_dir / "02_ibit_equity_curve_extended.png"
    plt.figure(figsize=(12, 6))
    for c, lab in [
        ("buy_hold", "Buy & Hold"),
        ("amma", "AMMA"),
        ("trend_ou_mining", "Trend+OU+Mining"),
        ("ou_zscore", "OU+ZScore"),
        ("stochastic_tactical", "Stochastic Tactical"),
        ("grand_all", "Grand All"),
    ]:
        plt.plot(equity.index, equity[c], label=lab, linewidth=1.8)

    plt.title("IBIT Equity Curves (Extended, No Leverage, Cost-Aware)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close()

    print(f"Saved core plot: {out1}")
    print(f"Saved extended plot: {out2}")


if __name__ == "__main__":
    main()
