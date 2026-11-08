import argparse
import logging
import time

import numpy as np
import pandas as pd

from src.features import ROOT, build_panel
from src.metrics import score
from src.models import MODELS

RESULTS = ROOT / "results"
PARTIAL = RESULTS / "partial"
H = 168
SEASON = 168

log = logging.getLogger("backtest")


def origins(panel: pd.DataFrame, n: int, step: int) -> list[pd.Timestamp]:
    last = panel["ds"].max() - pd.Timedelta(hours=H)
    return [last - pd.Timedelta(hours=i * step) for i in reversed(range(n))]


def mase_scale(train: pd.DataFrame) -> pd.Series:
    y = train.set_index("ds").groupby("unique_id")["y"]
    return y.apply(lambda s: np.mean(np.abs(s.diff(SEASON).dropna()))).rename("scale")


def run_model(name: str, panel: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    actuals = panel[["unique_id", "ds", "y", "imputed"]]
    out = []
    t_model = time.perf_counter()
    for i, origin in enumerate(cutoffs, 1):
        train = panel[panel["ds"] <= origin]
        future = panel[(panel["ds"] > origin) & (panel["ds"] <= origin + pd.Timedelta(hours=H))]
        future = future.drop(columns=["y", "imputed"])
        scale = mase_scale(train)
        log.debug("train %s bis %s, %d Zeilen | future %d Zeilen", train["ds"].min(), origin, len(train), len(future))
        log.debug("MASE-Skala: %s", scale.round(1).to_dict())

        t = time.perf_counter()
        fc = MODELS[name](train, future)
        n_raw = len(fc)
        fc = fc.merge(actuals, on=["unique_id", "ds"]).merge(scale, on="unique_id").assign(origin=origin)
        if len(fc) != n_raw:
            log.warning("%s %s: %d von %d Zeilen nach Merge verloren", name, origin, n_raw - len(fc), n_raw)
        nans = fc[["yhat", "lo", "hi"]].isna().sum()
        if nans.any():
            log.warning("%s %s: NaN in Prognose %s", name, origin, nans.to_dict())
        log.debug("Zeilen pro Horizont: %s", fc.groupby("horizon").size().to_dict())

        week = fc[(fc["horizon"] == H) & ~fc["imputed"]]
        mase = ((week["y"] - week["yhat"]).abs() / week["scale"]).mean()
        log.info(
            "%-12s [%2d/%d] %s  %5.1fs  MASE(7d) %.3f",
            name, i, len(cutoffs), f"{origin + pd.Timedelta(hours=1):%Y-%m-%d}", time.perf_counter() - t, mase,
        )
        out.append(fc)
    log.info("%s fertig in %.1f min", name, (time.perf_counter() - t_model) / 60)
    return pd.concat(out, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origins", type=int, default=45)
    parser.add_argument("--step", type=int, default=192, help="hours between forecast origins")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--resume", action="store_true", help="reuse checkpoints in results/partial/")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("cmdstanpy", "prophet", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    PARTIAL.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    cutoffs = origins(panel, args.origins, args.step)
    log.info(
        "Panel: %d Sensoren, %s bis %s, %d imputierte Stunden",
        panel["unique_id"].nunique(), panel["ds"].min(), panel["ds"].max(), panel["imputed"].sum(),
    )
    log.info("Sensoren: %s", ", ".join(sorted(panel["unique_id"].unique())))
    log.info("%d Ursprünge von %s bis %s, Modelle: %s", len(cutoffs), cutoffs[0], cutoffs[-1], args.models)

    parts = []
    for name in args.models:
        checkpoint = PARTIAL / f"{name}.parquet"
        if args.resume and checkpoint.exists():
            log.info("%s: Checkpoint gefunden, übersprungen", name)
            parts.append(pd.read_parquet(checkpoint))
            continue
        fc = run_model(name, panel, cutoffs)
        fc.to_parquet(checkpoint, index=False)
        log.info("Checkpoint gespeichert: %s", checkpoint.relative_to(ROOT))
        parts.append(fc)

    cols = ["model", "horizon", "unique_id", "origin", "ds", "y", "yhat", "lo", "hi", "imputed", "scale"]
    fc = pd.concat(parts, ignore_index=True)[cols]
    fc[["yhat", "lo", "hi"]] = fc[["yhat", "lo", "hi"]].clip(lower=0)
    fc = fc.astype({c: "float32" for c in ("y", "yhat", "lo", "hi", "scale")})

    path = RESULTS / "forecasts.parquet"
    if path.exists() and set(args.models) != set(MODELS):
        kept = pd.read_parquet(path)
        kept = kept[~kept["model"].astype(str).isin(fc["model"].unique())]
        fc = pd.concat([kept.astype({"model": str, "unique_id": str}), fc], ignore_index=True)
    fc.astype({"model": "category", "unique_id": "category"}).to_parquet(path, index=False)
    metrics = score(fc)
    metrics.to_parquet(RESULTS / "metrics.parquet", index=False)
    panel[["unique_id", "ds", "y", "imputed", "holiday"]].astype({"y": "float32"}).to_parquet(
        RESULTS / "history.parquet", index=False
    )

    board = metrics.groupby(["horizon", "model"])[["MASE", "sMAPE", "Coverage80"]].mean().round(3)
    log.info("Ergebnisse gespeichert in %s\n%s", RESULTS.relative_to(ROOT), board.to_string())


if __name__ == "__main__":
    main()
