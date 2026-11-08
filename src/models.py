import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS, SeasonalNaive

from src.features import holiday_dates

log = logging.getLogger("models")

LEVEL = 80
Q_LO, Q_HI = 0.1, 0.9
HORIZONS = (24, 168)


def _expand(fc: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Horizon-agnostic models: the 24h forecast is the first day of the 168h forecast."""
    fc = fc.sort_values(["model", "unique_id", "ds"])
    step = fc.groupby(["model", "unique_id"]).cumcount() + 1
    return pd.concat([fc[step <= h].assign(horizon=h) for h in horizons], ignore_index=True)


def statistical(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    h = future.groupby("unique_id").size().iloc[0]
    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=168, alias="SeasonalNaive"),
            MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"), alias="MSTL-ETS"),
        ],
        freq="h",
        n_jobs=-1,
    )
    fc = sf.forecast(df=train[["unique_id", "ds"]].assign(y=np.log1p(train["y"])), h=h, level=[LEVEL])
    out = []
    for name in ("SeasonalNaive", "MSTL-ETS"):
        out.append(
            fc[["unique_id", "ds"]].assign(
                model=name,
                yhat=np.expm1(fc[name]),
                lo=np.expm1(fc[f"{name}-lo-{LEVEL}"]),
                hi=np.expm1(fc[f"{name}-hi-{LEVEL}"]),
            )
        )
    return _expand(pd.concat(out))


def _prophet_one(train: pd.DataFrame, future: pd.DataFrame, hol: pd.DataFrame) -> pd.DataFrame:
    from prophet import Prophet

    stan_log = logging.getLogger("cmdstanpy")
    stan_log.addHandler(logging.NullHandler())
    stan_log.propagate = False
    stan_log.setLevel(logging.WARNING)
    m = Prophet(
        daily_seasonality=False,
        holidays=hol,
        interval_width=LEVEL / 100,
        uncertainty_samples=300,
    )
    m.add_seasonality("daily_workday", period=1, fourier_order=10, condition_name="workday")
    m.add_seasonality("daily_offday", period=1, fourier_order=10, condition_name="offday")
    m.add_regressor("temp")
    m.add_regressor("precip")
    cols = ["ds", "temp", "precip", "offday"]
    fit = train[cols].assign(y=np.log1p(train["y"]), workday=~train["offday"])
    m.fit(fit)
    pred = m.predict(future[cols].assign(workday=~future["offday"]))
    return pd.DataFrame(
        {
            "unique_id": future["unique_id"].to_numpy(),
            "ds": pred["ds"].to_numpy(),
            "model": "Prophet",
            "yhat": np.expm1(pred["yhat"]).to_numpy(),
            "lo": np.expm1(pred["yhat_lower"]).to_numpy(),
            "hi": np.expm1(pred["yhat_upper"]).to_numpy(),
        }
    )


def prophet(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    hol = holiday_dates(range(train["ds"].dt.year.min(), future["ds"].dt.year.max() + 1))
    fcs = Parallel(n_jobs=-1)(
        delayed(_prophet_one)(train[train["unique_id"] == uid], fut, hol)
        for uid, fut in future.groupby("unique_id")
    )
    return _expand(pd.concat(fcs))


LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
}
NUM_ROUNDS = 500


def lag_features(df: pd.DataFrame, h: int) -> tuple[pd.DataFrame, list[str]]:
    g = df.groupby("unique_id")["y"]
    lags = [lag for lag in (24, 48, 72, 168, 336, 504, 672) if lag >= h]
    feats = {f"lag_{lag}": g.shift(lag) for lag in lags}
    feats["same_hour_4w"] = sum(g.shift(lag) for lag in (168, 336, 504, 672)) / 4
    shifted = g.shift(h)
    for w in (24, 168):
        feats[f"mean_{w}"] = shifted.groupby(df["unique_id"]).transform(lambda s: s.rolling(w).mean())
    return df.assign(**feats), list(feats)


def lightgbm(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    import lightgbm as lgb

    df = pd.concat([train, future.assign(y=np.nan, imputed=False)], ignore_index=True)
    df = df.sort_values(["unique_id", "ds"], ignore_index=True)
    df["y"] = np.log1p(df["y"])
    df["sensor"] = df["unique_id"].astype("category")
    df["step"] = df.groupby("unique_id").cumcount() - df.groupby("unique_id")["y"].transform("count") + 1
    base = ["sensor", "hour", "dow", "doy", "holiday", "offday", "temp", "precip"]

    out = []
    for h in HORIZONS:
        data, lag_cols = lag_features(df, h)
        cols = base + lag_cols
        fit = data[(data["step"] <= 0) & ~data["imputed"]].dropna(subset=lag_cols)
        pred = data[(data["step"] >= 1) & (data["step"] <= h)]
        log.debug("LightGBM h=%d: %d Trainingszeilen, %d Features, %d Prognosezeilen", h, len(fit), len(cols), len(pred))
        res = pred[["unique_id", "ds"]].assign(model="LightGBM", horizon=h)
        for col, objective in (("yhat", {"objective": "l2"}),
                               ("lo", {"objective": "quantile", "alpha": Q_LO}),
                               ("hi", {"objective": "quantile", "alpha": Q_HI})):
            booster = lgb.train({**LGB_PARAMS, **objective}, lgb.Dataset(fit[cols], fit["y"]), NUM_ROUNDS)
            res[col] = np.expm1(booster.predict(pred[cols]))
        out.append(res)
    return pd.concat(out, ignore_index=True)


CHRONOS_MODEL = "amazon/chronos-bolt-base"
CHRONOS_CONTEXT = 2048


@cache
def _chronos_pipeline():
    import torch
    from chronos import BaseChronosPipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map=device, torch_dtype=torch.float32)


def _chronos_predict(context: list[np.ndarray], h: int) -> tuple[np.ndarray, str]:
    import warnings

    import torch

    warnings.filterwarnings("ignore", message="We recommend keeping prediction length")
    pipeline = _chronos_pipeline()
    quantiles, _ = pipeline.predict_quantiles(
        [torch.tensor(c, dtype=torch.float32) for c in context],
        prediction_length=h,
        quantile_levels=[Q_LO, 0.5, Q_HI],
    )
    return quantiles.cpu().numpy(), str(pipeline.model.device)


@cache
def _chronos_worker() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))


def chronos(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    h = future.groupby("unique_id").size().iloc[0]
    ids = sorted(future["unique_id"].unique())
    context = [train.loc[train["unique_id"] == uid, "y"].to_numpy()[-CHRONOS_CONTEXT:] for uid in ids]
    log.debug("Chronos Kontext: %s Stunden je Sensor, h=%d", [len(c) for c in context], h)
    q, device = _chronos_worker().submit(_chronos_predict, context, h).result(timeout=600)
    log.debug("Chronos lief auf %s, Output %s", device, q.shape)
    fut = future.sort_values(["unique_id", "ds"])
    fc = fut[["unique_id", "ds"]].assign(
        model="Chronos-Bolt",
        yhat=q[:, :, 1].ravel(),
        lo=q[:, :, 0].ravel(),
        hi=q[:, :, 2].ravel(),
    )
    return _expand(fc)


MODELS = {
    "statistical": statistical,
    "prophet": prophet,
    "lightgbm": lightgbm,
    "chronos": chronos,
}
