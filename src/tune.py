"""Tune LightGBM with Optuna on a validation window that ends before the first backtest origin.

    python -m src.tune --trials 50
    python -m src.tune --trials 50 --debug

Writes results/lgb_params.json (used by the `lightgbm_tuned` model) and results/tuning.json (ablation).
"""

import argparse
import json
import logging
import time

import numpy as np
import optuna
import pandas as pd

from src import tracking
from src.backtest import N_ORIGINS, RESULTS, SEASON, STEP, origins
from src.features import build_panel, lgb_features
from src.models import HORIZONS, LGB_TUNED, LGB_V1, lgb_fit, lgb_predict, lgb_prepare

VAL_WEEKS = 10

log = logging.getLogger("tune")


def windows(panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    test_start = origins(panel, N_ORIGINS, STEP)[0]
    return test_start - pd.Timedelta(weeks=VAL_WEEKS), test_start


def split(panel: pd.DataFrame, h: int, extended: bool):
    """Lags >= h make a row-wise holdout equivalent to forecasting it h steps ahead."""
    val_start, test_start = windows(panel)
    data, cols, target_cols = lgb_features(lgb_prepare(panel), h, extended)
    data = data[~data["imputed"]].dropna(subset=target_cols)
    fit = data[data["ds"] <= val_start]
    val = data[(data["ds"] > val_start) & (data["ds"] <= test_start)]
    log.debug("h=%d extended=%s: fit %d Zeilen, val %d Zeilen, %d Features", h, extended, len(fit), len(val), len(cols))
    return fit, val, cols


def mase_scale(panel: pd.DataFrame) -> pd.Series:
    train = panel[panel["ds"] <= windows(panel)[0]].set_index("ds").groupby("unique_id")["y"]
    return train.apply(lambda s: np.mean(np.abs(s.diff(SEASON).dropna())))


def val_mase(val: pd.DataFrame, yhat: np.ndarray, scale: pd.Series) -> float:
    err = pd.Series(np.abs(np.expm1(val["y"].to_numpy()) - yhat), index=val["unique_id"].to_numpy())
    return float((err.groupby(level=0).mean() / scale).mean())


def evaluate(fit, val, cols, cfg, scale) -> float:
    return val_mase(val, lgb_predict(lgb_fit(fit, cols, cfg), val, cols, cfg), scale)


def tune_horizon(panel: pd.DataFrame, h: int, trials: int, scale: pd.Series, run) -> dict:
    fit, val, cols = split(panel, h, extended=True)
    run.define_metric(f"h{h}/*", step_metric=f"h{h}/trial")

    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "objective": trial.suggest_categorical("objective", ["l2", "l1", "huber"]),
            "target": trial.suggest_categorical("target", ["log", "residual"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        }
        t = time.perf_counter()
        booster = lgb_fit(fit, cols, cfg, valid=val)
        trial.set_user_attr("num_rounds", booster.best_iteration)
        mase = val_mase(val, lgb_predict(booster, val, cols, cfg), scale)
        best = min([mase, *(t.value for t in trial.study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))])
        run.log({f"h{h}/trial": trial.number + 1, f"h{h}/val_mase": mase, f"h{h}/best_mase": best,
                 f"h{h}/rounds": booster.best_iteration, **{f"h{h}/{k}": v for k, v in cfg.items()}})
        log.debug("Trial %d Parameter: %s", trial.number, cfg)
        log.info(
            "h=%-3d Trial %2d/%d  MASE %.4f  %-5s %-8s lr %.3f leaves %3d rounds %4d  %5.1fs",
            h, trial.number + 1, trials, mase, cfg["objective"], cfg["target"],
            cfg["learning_rate"], cfg["num_leaves"], booster.best_iteration, time.perf_counter() - t,
        )
        return mase

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    cfg = best.params | {"subsample_freq": 1, "num_rounds": best.user_attrs["num_rounds"]}
    log.info("h=%d bestes Trial %d: MASE %.4f  %s", h, best.number + 1, best.value, cfg)
    return cfg


def ablation(panel: pd.DataFrame, h: int, tuned: dict, scale: pd.Series) -> dict:
    base = split(panel, h, extended=False)
    ext = split(panel, h, extended=True)
    result = {
        "v1": evaluate(*base, LGB_V1, scale),
        "v1 + Features": evaluate(*ext, LGB_V1, scale),
        "Tuning": evaluate(*base, tuned, scale),
        "Tuning + Features": evaluate(*ext, tuned, scale),
    }
    log.info("h=%d Ablation (Validierungs-MASE): %s", h, {k: round(v, 4) for k, v in result.items()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    panel = build_panel()
    val_start, test_start = windows(panel)
    scale = mase_scale(panel)
    log.info("Training bis %s, Validierung bis %s (Backtest beginnt danach)", val_start, test_start)
    log.debug("MASE-Skala: %s", scale.round(1).to_dict())

    config = {"trials": args.trials, "val_weeks": VAL_WEEKS, "val_start": str(val_start), "test_start": str(test_start)}
    with tracking.start("tune", not args.no_wandb, config=config) as run:
        params, report = {}, {}
        for h in HORIZONS:
            params[h] = tune_horizon(panel, h, args.trials, scale, run)
            report[h] = ablation(panel, h, params[h], scale)
            run.summary[f"h{h}/best"] = params[h]

        LGB_TUNED.write_text(json.dumps(params, indent=2))
        (RESULTS / "tuning.json").write_text(json.dumps(report, indent=2))
        ablation_table = pd.DataFrame(report).rename(columns=lambda h: f"MASE h={h}").rename_axis("variant")
        run.log({"ablation": tracking.table(ablation_table)})
        tracking.log_file(run, LGB_TUNED, "lgb-params", "params")
    log.info("Gespeichert: %s, %s", LGB_TUNED.name, "tuning.json")


if __name__ == "__main__":
    main()
