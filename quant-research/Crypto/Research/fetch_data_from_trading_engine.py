"""Fetch data-engineering prices/model-state from trading-engine and export for quant-research.

Usage (from repo root):
  python quant-research/Crypto/Research/fetch_data_from_trading_engine.py \
    --start-date 2021-01-01 \
    --end-date 2025-01-01 \
    --out-dir quant-research/Crypto/Data

Notes:
- Requires cloud access/env that `trading_engine.core.read_data()` expects.
- Exports:
  - prices_wide.csv            (date + universe columns)
  - model_state_snapshot.csv   (date,ticker,adjusted_close_1d + selected features)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--out-dir", default="quant-research/Crypto/Data")
    p.add_argument(
        "--universe",
        nargs="*",
        default=["SPY-US", "GLD-US", "SLV-US", "TLT-US", "IBIT-US", "ETHA-US"],
    )
    p.add_argument(
        "--features",
        nargs="*",
        default=["close_momentum_10", "close_momentum_20", "close_momentum_60"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]

    te_src = root / "quant-research" / "trading-engine" / "src"
    if str(te_src) not in sys.path:
        sys.path.insert(0, str(te_src))

    import polars as pl
    from trading_engine.core import read_data, create_model_state, calculate_max_lookback

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading raw data from data-engineering source via trading_engine.core.read_data() ...")
    raw_lf = read_data(include_supplemental=False)

    lookback_days = calculate_max_lookback(
        features=args.features,
        models=[],
        aggregators=[],
        optimizers=[],
    )
    print(f"Computed lookback days: {lookback_days}")

    model_state, prices = create_model_state(
        lf=raw_lf,
        features=args.features,
        start_date=start_date,
        end_date=end_date,
        universe=args.universe,
        total_lookback_days=lookback_days,
        return_bundle=False,
    )

    # Keep only analysis horizon for exports
    ms = model_state.filter(pl.col("date").is_between(start_date, end_date))
    px = prices.filter(pl.col("date").is_between(start_date, end_date))

    price_csv = out_dir / "prices_wide.csv"
    ms_csv = out_dir / "model_state_snapshot.csv"

    px.write_csv(price_csv)

    keep_cols = ["date", "ticker", "adjusted_close_1d"]
    keep_cols.extend([c for c in args.features if c in ms.columns])
    ms.select([c for c in keep_cols if c in ms.columns]).write_csv(ms_csv)

    print(f"Wrote: {price_csv}")
    print(f"Wrote: {ms_csv}")


if __name__ == "__main__":
    main()
