import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    minp = min(window, max(3, window // 3))
    mu = series.rolling(window, min_periods=minp).mean()
    sd = series.rolling(window, min_periods=minp).std()
    z = (series - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def compute_mining_edge(
    df: pd.DataFrame,
    price_column: str = "IBIT_close",
    cost_column: str = "COST_TO_MINE",
) -> pd.Series:
    """
    Edge = log(price / cost)
    """
    price = df[price_column].astype(float)
    cost = df[cost_column].astype(float)
    edge = np.log(price / cost)
    return edge.replace([np.inf, -np.inf], np.nan)


def compute_mining_margin(
    df: pd.DataFrame,
    price_column: str = "IBIT_close",
    cost_column: str = "COST_TO_MINE",
) -> pd.Series:
    """
    Margin = (price - cost) / cost
    """
    price = df[price_column].astype(float)
    cost = df[cost_column].astype(float)
    margin = (price - cost) / cost
    return margin.replace([np.inf, -np.inf], np.nan)


def mining_signal(
    df: pd.DataFrame,
    price_column: str = "IBIT_close",
    cost_column: str = "COST_TO_MINE",
    z_window: int = 180,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    use_log_edge: bool = True,
) -> pd.Series:
    """
    Long-only mining-cost regime.

    Uses a standardized edge (z-score of log(price/cost) or margin).
    We interpret *very cheap vs cost* as a buy signal.

    Trade:
      - Enter long when z < -entry_z
      - Exit when z > -exit_z

    Output is UN-SHIFTED target position in [0, 1].
    """
    if use_log_edge:
        edge = compute_mining_edge(df, price_column=price_column, cost_column=cost_column)
    else:
        edge = compute_mining_margin(df, price_column=price_column, cost_column=cost_column)

    z = rolling_zscore(edge, z_window)

    pos = pd.Series(0.0, index=df.index)
    in_pos = 0.0

    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = in_pos
            continue

        if in_pos == 0.0:
            if zi < -float(entry_z):
                in_pos = 1.0
        else:
            if zi > -float(exit_z):
                in_pos = 0.0

        pos.iloc[i] = in_pos

    return pos.fillna(0.0).clip(0.0, 1.0)