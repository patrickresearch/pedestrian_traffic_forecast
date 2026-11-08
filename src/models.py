import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS, SeasonalNaive

from src.features import ROOT, holiday_dates, lgb_features

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


LGB_V1 = {
    "objective": "l2",
    "target": "log",
    "num_rounds": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}
LGB_TUNED = ROOT / "results" / "lgb_params.json"
LGB_MAX_ROUNDS = 2000


def lgb_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["unique_id", "ds"], ignore_index=True)
    return df.assign(y=np.log1p(df["y"]), sensor=df["unique_id"].astype("category"))


def lgb_target(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """`target: residual` learns log(y) minus the 4-week same-hour mean instead of log(y)."""
    return df["y"] - df["same_hour_4w"] if cfg["target"] == "residual" else df["y"]


def lgb_fit(fit: pd.DataFrame, cols: list[str], cfg: dict, valid: pd.DataFrame | None = None, **override):
    """With `valid`, rounds are chosen by early stopping instead of `num_rounds`."""
    import lightgbm as lgb

    params = {k: v for k, v in cfg.items() if k not in ("num_rounds", "target")} | override | {"verbose": -1}
    train_set = lgb.Dataset(fit[cols], lgb_target(fit, cfg))
    if valid is None:
        return lgb.train(params, train_set, cfg["num_rounds"])
    valid_set = lgb.Dataset(valid[cols], lgb_target(valid, cfg), reference=train_set)
    return lgb.train(
        params | {"metric": "l1"},
        train_set,
        LGB_MAX_ROUNDS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def lgb_predict(booster, pred: pd.DataFrame, cols: list[str], cfg: dict) -> np.ndarray:
    out = booster.predict(pred[cols])
    if cfg["target"] == "residual":
        out = out + pred["same_hour_4w"].to_numpy()
    return np.expm1(out)


def lgb_config(tuned: bool) -> dict[int, dict]:
    if not tuned:
        return {h: LGB_V1 for h in HORIZONS}
    if not LGB_TUNED.exists():
        raise FileNotFoundError(f"{LGB_TUNED} fehlt, zuerst `python -m src.tune` laufen lassen")
    return {int(h): cfg for h, cfg in json.loads(LGB_TUNED.read_text()).items()}


def lightgbm(train: pd.DataFrame, future: pd.DataFrame, tuned: bool = False) -> pd.DataFrame:
    name = "LightGBM-tuned" if tuned else "LightGBM"
    configs = lgb_config(tuned)
    df = lgb_prepare(pd.concat([train, future.assign(y=np.nan, imputed=False)], ignore_index=True))
    df["step"] = df.groupby("unique_id").cumcount() - df.groupby("unique_id")["y"].transform("count") + 1

    out = []
    for h in HORIZONS:
        cfg = configs[h]
        data, cols, target_cols = lgb_features(df, h, extended=tuned)
        fit = data[(data["step"] <= 0) & ~data["imputed"]].dropna(subset=target_cols)
        pred = data[(data["step"] >= 1) & (data["step"] <= h)]
        log.debug("%s h=%d: %d Trainingszeilen, %d Features, %d Prognosezeilen", name, h, len(fit), len(cols), len(pred))
        res = pred[["unique_id", "ds"]].assign(model=name, horizon=h)
        res["yhat"] = lgb_predict(lgb_fit(fit, cols, cfg), pred, cols, cfg)
        res["lo"] = lgb_predict(lgb_fit(fit, cols, cfg, objective="quantile", alpha=Q_LO), pred, cols, cfg)
        res["hi"] = lgb_predict(lgb_fit(fit, cols, cfg, objective="quantile", alpha=Q_HI), pred, cols, cfg)
        out.append(res)
    return pd.concat(out, ignore_index=True)


CHRONOS_BOLT = "amazon/chronos-bolt-base"
CHRONOS_2 = "amazon/chronos-2"
BOLT_CONTEXT = 2048
CHRONOS2_CONTEXT = 8192
COVARIATES = ["offday", "holiday", "temp", "precip"]


@cache
def _chronos_pipeline(model_id: str):
    import torch
    from chronos import BaseChronosPipeline
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = snapshot_download(model_id, local_files_only=True)
    except LocalEntryNotFoundError:
        path = snapshot_download(model_id)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return BaseChronosPipeline.from_pretrained(path, device_map=device, torch_dtype=torch.float32)


def _bolt_predict(context: list[np.ndarray], h: int) -> tuple[np.ndarray, str]:
    import warnings

    import torch

    warnings.filterwarnings("ignore", message="We recommend keeping prediction length")
    pipeline = _chronos_pipeline(CHRONOS_BOLT)
    quantiles, _ = pipeline.predict_quantiles(
        [torch.tensor(c, dtype=torch.float32) for c in context],
        prediction_length=h,
        quantile_levels=[Q_LO, 0.5, Q_HI],
    )
    return quantiles.cpu().numpy(), str(pipeline.model.device)


def _chronos2_predict(context: pd.DataFrame, future: pd.DataFrame | None, h: int) -> tuple[pd.DataFrame, str]:
    pipeline = _chronos_pipeline(CHRONOS_2)
    pred = pipeline.predict_df(
        context,
        future_df=future,
        id_column="unique_id",
        timestamp_column="ds",
        target="y",
        prediction_length=h,
        quantile_levels=[Q_LO, 0.5, Q_HI],
        freq="h",
    )
    return pred, str(pipeline.model.device)


@cache
def _chronos_worker() -> ProcessPoolExecutor:
    """torch and LightGBM load different OpenMP runtimes that deadlock in one process."""
    return ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))


def chronos(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    h = future.groupby("unique_id").size().iloc[0]
    ids = sorted(future["unique_id"].unique())
    context = [train.loc[train["unique_id"] == uid, "y"].to_numpy()[-BOLT_CONTEXT:] for uid in ids]
    log.debug("Chronos-Bolt Kontext: %s Stunden je Sensor, h=%d", [len(c) for c in context], h)
    q, device = _chronos_worker().submit(_bolt_predict, context, h).result(timeout=600)
    log.debug("Chronos-Bolt lief auf %s, Output %s", device, q.shape)
    fut = future.sort_values(["unique_id", "ds"])
    fc = fut[["unique_id", "ds"]].assign(
        model="Chronos-Bolt",
        yhat=q[:, :, 1].ravel(),
        lo=q[:, :, 0].ravel(),
        hi=q[:, :, 2].ravel(),
    )
    return _expand(fc)


def chronos2(train: pd.DataFrame, future: pd.DataFrame, covariates: bool = False) -> pd.DataFrame:
    h = future.groupby("unique_id").size().iloc[0]
    cov = COVARIATES if covariates else []
    context = (
        train[["unique_id", "ds", "y", *cov]]
        .groupby("unique_id")
        .tail(CHRONOS2_CONTEXT)
        .astype({c: float for c in cov})
        .sort_values(["unique_id", "ds"])
    )
    fut = future[["unique_id", "ds", *cov]].astype({c: float for c in cov}).sort_values(["unique_id", "ds"])
    log.debug("Chronos-2 Kontext: %d Zeilen, Kovariaten %s, h=%d", len(context), cov, h)
    pred, device = _chronos_worker().submit(_chronos2_predict, context, fut if cov else None, h).result(timeout=900)
    log.debug("Chronos-2 lief auf %s, Output %s", device, pred.shape)
    fc = pred[["unique_id", "ds"]].assign(
        model="Chronos-2-Cov" if covariates else "Chronos-2",
        yhat=pred[str(0.5)],
        lo=pred[str(Q_LO)],
        hi=pred[str(Q_HI)],
    )
    return _expand(fc)


MODELS = {
    "statistical": statistical,
    "prophet": prophet,
    "lightgbm": lightgbm,
    "lightgbm_tuned": partial(lightgbm, tuned=True),
    "chronos": chronos,
    "chronos2": chronos2,
    "chronos2_cov": partial(chronos2, covariates=True),
}
