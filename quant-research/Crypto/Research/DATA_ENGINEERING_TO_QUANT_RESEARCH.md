# Data Engineering → Quant Research (Practical Workflow)

This explains how to fetch data using the `trading-engine` data pipeline and use it in `quant-research`.

## 1) What to run

From repository root:

```bash
python quant-research/Crypto/Research/fetch_data_from_trading_engine.py \
  --start-date 2021-01-01 \
  --end-date 2025-01-01 \
  --out-dir quant-research/Crypto/Data \
  --universe SPY-US GLD-US SLV-US TLT-US IBIT-US ETHA-US \
  --features close_momentum_10 close_momentum_20 close_momentum_60
```

This uses:
- `trading_engine.core.read_data()` to pull raw data from the data-engineering source.
- `trading_engine.core.create_model_state()` to build model-state and prices over your universe/date window.

## 2) Output files

The script writes:
- `quant-research/Crypto/Data/prices_wide.csv`
- `quant-research/Crypto/Data/model_state_snapshot.csv`

## 3) Use in quant-research notebooks/scripts

Typical usage:

```python
import pandas as pd

prices = pd.read_csv("quant-research/Crypto/Data/prices_wide.csv", parse_dates=["date"])
model_state = pd.read_csv("quant-research/Crypto/Data/model_state_snapshot.csv", parse_dates=["date"])

# Example: IBIT close from wide prices
ibit = prices[["date", "IBIT-US"]].dropna().rename(columns={"IBIT-US": "close"})
```

## 4) Why this is the correct bridge

- You are no longer mixing separate ad-hoc datasets.
- Universe/date/feature logic comes from one source (`trading_engine.core`) and can be reused by both portfolio research and IBIT strategy studies.

## 5) Troubleshooting

- If cloud/GCS auth is missing, `read_data()` will fail. Run in the same authenticated environment as trading-engine simulations.
- If a ticker is absent in raw source for the period, that column can be null/empty in `prices_wide.csv`.
