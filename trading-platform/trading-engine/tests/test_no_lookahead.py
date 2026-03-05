import numpy as np
import pandas as pd

from trading_engine.models.catalogue.ibit_grand_stack import generate_positions


def test_no_lookahead():
    dates = pd.date_range("2022-01-01", periods=320, freq="D")
    px = pd.Series(np.linspace(100, 150, len(dates)) + np.sin(np.arange(len(dates))), index=dates)
    df = pd.DataFrame({"IBIT_close": px, "COST_TO_MINE": px * 0.8}, index=dates)

    out = generate_positions(df)
    pos = out["IBIT-US"]

    assert pos.iloc[0] == 0
    assert pos.isna().sum() == 0

    df_changed = df.copy()
    df_changed.loc[df_changed.index[-1], "IBIT_close"] = df_changed.loc[df_changed.index[-1], "IBIT_close"] * 2.0

    out_changed = generate_positions(df_changed)
    pos_changed = out_changed["IBIT-US"]

    pd.testing.assert_series_equal(pos.iloc[:-1], pos_changed.iloc[:-1], check_names=False)
