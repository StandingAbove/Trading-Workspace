import pandas as pd

from . import amma


def generate_positions(data: pd.DataFrame) -> pd.DataFrame:
    pos = amma.position(data)
    return pd.DataFrame({"date": data.index, "IBIT-US": pos})
