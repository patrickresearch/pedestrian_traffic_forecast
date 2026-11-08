import numpy as np
import pandas as pd


def mase(y, yhat, scale) -> float:
    return float(np.mean(np.abs(y - yhat)) / scale)


def smape(y, yhat) -> float:
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.divide(2 * np.abs(y - yhat), denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return float(100 * np.mean(ratio))


def coverage(y, lo, hi) -> float:
    return float(100 * np.mean((y >= lo) & (y <= hi)))


def score(fc: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "horizon", "unique_id", "origin"]
    return (
        fc[~fc["imputed"]]
        .groupby(keys)
        .apply(
            lambda g: pd.Series(
                {
                    "MASE": mase(g["y"], g["yhat"], g["scale"].iloc[0]),
                    "sMAPE": smape(g["y"].to_numpy(), g["yhat"].to_numpy()),
                    "Coverage80": coverage(g["y"], g["lo"], g["hi"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
