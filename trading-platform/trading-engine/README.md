# Trading Engine

## Run tests

```bash
cd trading-platform/trading-engine
PYTHONPATH=src pytest -q
```

## Run local simulation/backtest

```bash
cd trading-platform/trading-engine
python - <<'PY'
import pandas as pd
import numpy as np
from trading_engine.models.registry import get_model

idx = pd.date_range('2022-01-01', periods=400, freq='D')
px = pd.Series(np.linspace(100, 200, len(idx)) + 2*np.sin(np.arange(len(idx))/8), index=idx)
df = pd.DataFrame({'IBIT_close': px, 'COST_TO_MINE': px*0.78}, index=idx)

for name in ["IBIT_AMMA", "IBIT_GRAND_STACK"]:
    out = get_model(name)(df)
    print(name, out.tail(3))
PY
```

## Run dashboard pipeline

```bash
cd trading-platform/trading-engine
python - <<'PY'
from trading_engine.dashboard import SELECTABLE_MODELS
print("Selectable models:", SELECTABLE_MODELS)
PY
```
