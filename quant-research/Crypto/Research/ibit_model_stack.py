from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Backtest.amma import amma_from_ibit_csv
from Data.raw_data_loader import load_ibit_with_mining_cost
from Models.mining import mining_signal
from Models.ou import ou_signal
from Models.trend import trend_signal
from Models.zscore import zscore_signal


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    turnover: pd.Series
    position: pd.Series


def run_backtest(price: pd.Series, pos: pd.Series, fee_bps: float, slippage_bps: float) -> BacktestResult:
    price = price.astype(float)
    ret = price.pct_change().fillna(0.0)
    p = pos.reindex(price.index).fillna(0.0).clip(0.0, 1.0)
    turnover = p.diff().abs().fillna(p.abs())
    costs = turnover * ((fee_bps + slippage_bps) / 1e4)
    net = (p * ret) - costs
    eq = (1.0 + net).cumprod()
    return BacktestResult(net, eq, turnover, p)


def sharpe(x: pd.Series, periods: int = 252) -> float:
    x = x.dropna()
    if len(x) < 10:
        return np.nan
    s = x.std(ddof=1)
    if s <= 0 or not np.isfinite(s):
        return np.nan
    return float(np.sqrt(periods) * x.mean() / s)


def performance_table(ret_df: pd.DataFrame) -> pd.DataFrame:
    def cagr(r: pd.Series) -> float:
        eq = (1.0 + r.fillna(0.0)).cumprod()
        yrs = len(eq) / 252.0
        return float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan

    def sortino(r: pd.Series) -> float:
        dn = r[r < 0]
        if len(dn) < 2:
            return np.nan
        dsv = dn.std(ddof=1) * np.sqrt(252.0)
        return float((r.mean() * 252.0) / dsv) if dsv > 0 else np.nan

    def maxdd(r: pd.Series) -> float:
        eq = (1.0 + r.fillna(0.0)).cumprod()
        dd = eq / eq.cummax() - 1.0
        return float(dd.min())

    rows = []
    for c in ret_df.columns:
        r = ret_df[c].dropna()
        rows.append(
            {
                "Strategy": c,
                "CAGR": cagr(r),
                "Sharpe": sharpe(r),
                "Sortino": sortino(r),
                "MaxDD": maxdd(r),
                "Volatility": float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) > 1 else np.nan,
                "WinRate": float((r > 0).mean()) if len(r) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False)


def load_ibit_inputs(crypto_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel = load_ibit_with_mining_cost(
        str(crypto_root / "Data" / "IBIT_Data.csv"),
        str(crypto_root / "Data" / "cleaned_crypto_data.csv"),
        forward_fill_mining_cost=True,
    ).sort_index()
    panel["close"] = panel["close"].astype(float)

    model_df = pd.DataFrame(index=panel.index)
    model_df["BTC-USD_close"] = panel["close"]
    model_df["COST_TO_MINE"] = panel["mining_cost"]
    return panel, model_df


def _best_series_by_grid(price: pd.Series, candidates: Dict[str, pd.Series], fee_bps: float, slippage_bps: float) -> Tuple[str, pd.Series, float]:
    best_name = None
    best_series = None
    best_sharpe = -np.inf
    for name, pos in candidates.items():
        s = sharpe(run_backtest(price, pos, fee_bps, slippage_bps).returns)
        if np.isfinite(s) and s > best_sharpe:
            best_sharpe = float(s)
            best_name = name
            best_series = pos
    return best_name or "NA", best_series if best_series is not None else pd.Series(0.0, index=price.index), best_sharpe


def calibrate_model_positions(panel: pd.DataFrame, model_df: pd.DataFrame, fee_bps: float, slippage_bps: float) -> Dict[str, pd.Series]:
    price = panel["close"]

    # AMMA fixed schema
    amma = amma_from_ibit_csv(
        str((Path(__file__).resolve().parents[1] / "Data" / "IBIT_Data.csv")),
        momentum_weights={5: 0.25, 10: 0.25, 20: 0.25, 60: 0.25},
        threshold=0.0,
        long_enabled=True,
        short_enabled=False,
    ).rename(columns={"Date": "date"}).set_index("date")
    amma_pos = amma.reindex(price.index)["amma_position"].fillna(0.0).clip(0.0, 1.0)

    trend_candidates = {}
    for fw in [10, 20, 30]:
        for sw in [80, 100, 140]:
            if fw >= sw:
                continue
            for neutral in [0.0, 0.25, 0.5]:
                name = f"trend_fw{fw}_sw{sw}_n{neutral}"
                trend_candidates[name] = trend_signal(
                    model_df,
                    price_column="BTC-USD_close",
                    fast_window=fw,
                    slow_window=sw,
                    long_only=True,
                    leverage_aggressive=1.0,
                    leverage_neutral=neutral,
                    leverage_defensive=0.0,
                ).reindex(price.index).fillna(0.0).clip(0.0, 1.0)
    _, trend_pos, _ = _best_series_by_grid(price, trend_candidates, fee_bps, slippage_bps)

    mining_candidates = {}
    for zw in [90, 120, 180, 240]:
        for ez in [0.5, 0.8, 1.0]:
            for xz in [0.0, 0.1, 0.2]:
                name = f"mining_w{zw}_e{ez}_x{xz}"
                mining_candidates[name] = mining_signal(
                    model_df,
                    z_window=zw,
                    entry_z=ez,
                    exit_z=xz,
                    use_log_edge=True,
                ).reindex(price.index).fillna(0.0).clip(0.0, 1.0)
    _, mining_pos, _ = _best_series_by_grid(price, mining_candidates, fee_bps, slippage_bps)

    ou_candidates = {}
    for w in [60, 90, 120, 180]:
        for ez in [1.0, 1.25, 1.5]:
            for xz in [0.2, 0.3, 0.4]:
                name = f"ou_w{w}_e{ez}_x{xz}"
                ou_candidates[name] = ou_signal(
                    model_df["BTC-USD_close"],
                    window=w,
                    entry_z=ez,
                    exit_z=xz,
                    long_short=False,
                ).reindex(price.index).fillna(0.0).clip(0.0, 1.0)
    _, ou_pos, _ = _best_series_by_grid(price, ou_candidates, fee_bps, slippage_bps)

    z_candidates = {}
    for rw in [60, 90, 120, 180]:
        for ez in [1.0, 1.25, 1.5]:
            for xz in [0.2, 0.3, 0.4]:
                name = f"z_w{rw}_e{ez}_x{xz}"
                z_candidates[name] = zscore_signal(
                    model_df["BTC-USD_close"],
                    resid_window=rw,
                    entry_z=ez,
                    exit_z=xz,
                    long_short=False,
                    use_vol_target=False,
                    max_leverage=1.0,
                ).reindex(price.index).fillna(0.0).clip(0.0, 1.0)
    _, z_pos, _ = _best_series_by_grid(price, z_candidates, fee_bps, slippage_bps)

    return {"AMMA": amma_pos, "Trend": trend_pos, "Mining": mining_pos, "OU": ou_pos, "ZScore": z_pos}


def select_combo(panel: pd.DataFrame, positions: Dict[str, pd.Series], fee_bps: float, slippage_bps: float) -> Dict[str, object]:
    price = panel["close"]
    model_names = ["AMMA", "Trend", "Mining", "OU", "ZScore"]

    rets = {}
    for m in model_names:
        rets[m] = run_backtest(price, positions[m], fee_bps, slippage_bps).returns
    ret_df = pd.DataFrame(rets, index=price.index).dropna()

    # Static convex blend search
    rng = np.random.default_rng(20260228)
    best_static = {"name": "static", "weights": np.array([1, 0, 0, 0, 0], dtype=float), "returns": ret_df["AMMA"], "sharpe": sharpe(ret_df["AMMA"])}
    candidates = [np.eye(5)[i] for i in range(5)] + [np.ones(5) / 5]
    for _ in range(250_000):
        candidates.append(rng.dirichlet(np.ones(5)))

    for w in candidates:
        r = (ret_df[model_names] * w).sum(axis=1)
        s = sharpe(r)
        if np.isfinite(s) and s > best_static["sharpe"]:
            best_static = {"name": "static", "weights": np.array(w, dtype=float), "returns": r, "sharpe": float(s)}

    # Dynamic rolling-sharpe weighting
    lookback = 63
    roll = ret_df[model_names].rolling(lookback).apply(lambda x: 0.0 if x.std(ddof=1) == 0 else (x.mean() / x.std(ddof=1)), raw=False)
    roll = roll.clip(lower=0.0).fillna(0.0)
    denom = roll.sum(axis=1).replace(0.0, np.nan)
    dyn_w = roll.div(denom, axis=0).fillna(1.0 / len(model_names))
    dyn_r = (ret_df[model_names] * dyn_w).sum(axis=1)
    best_dynamic = {"name": "dynamic_roll_sharpe", "weights": dyn_w.mean(axis=0).values, "returns": dyn_r, "sharpe": sharpe(dyn_r)}

    amma_s = sharpe(ret_df["AMMA"])
    best = best_static if best_static["sharpe"] >= best_dynamic["sharpe"] else best_dynamic

    return {
        "model_names": model_names,
        "ret_df": ret_df,
        "amma_sharpe": amma_s,
        "selected": best,
        "static": best_static,
        "dynamic": best_dynamic,
    }


def discrepancy_explanation() -> str:
    return (
        "Research-02 grand combo Sharpe can be higher than IBIT notebook combo Sharpe because the two workflows were not on the same target domain. "
        "Research-02 combo was calibrated on BTC regime/return structure, then transferred to IBIT; transfer changes date coverage, turnover profile, cost drag, and signal scaling. "
        "If combo weights/params are selected on BTC but evaluated on IBIT, outperformance over AMMA is not guaranteed. "
        "Fix: calibrate all component models and combo weights directly on IBIT with consistent costs/no-leverage constraints and compare on the same sample."
    )
