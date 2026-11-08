from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
N_SENSORS = 8
MIN_COVERAGE = 0.99


def load_counts() -> pd.DataFrame:
    df = pd.read_parquet(
        RAW / "counts.parquet",
        columns=["location_id", "sensor_name", "sensing_date", "hourday", "pedestriancount"],
    )
    df["ds"] = pd.to_datetime(df["sensing_date"]) + pd.to_timedelta(df["hourday"], unit="h")
    end = df["ds"].max().normalize()
    return df.loc[df["ds"] < end].rename(columns={"pedestriancount": "y"})[
        ["location_id", "sensor_name", "ds", "y"]
    ]


def select_sensors(counts: pd.DataFrame, n: int = N_SENSORS) -> list[int]:
    hours = (counts["ds"].max() - counts["ds"].min()) / pd.Timedelta(hours=1) + 1
    stats = counts.groupby(["location_id", "sensor_name"])["y"].agg(["size", "mean"]).reset_index()
    stats = stats[stats["size"] / hours >= MIN_COVERAGE].sort_values("mean", ascending=False)
    return stats.drop_duplicates("sensor_name").head(n)["location_id"].tolist()


def make_panel(counts: pd.DataFrame, location_ids: list[int]) -> pd.DataFrame:
    """Complete hourly grid per sensor. Gaps are filled, but flagged as `imputed`."""
    df = counts[counts["location_id"].isin(location_ids)]
    names = df.drop_duplicates("location_id").set_index("location_id")["sensor_name"]
    grid = pd.date_range(df["ds"].min(), df["ds"].max(), freq="h")
    panel = (
        df.set_index(["location_id", "ds"])["y"]
        .astype(float)
        .unstack(0)
        .reindex(grid)
        .rename_axis("ds")
    )
    imputed = panel.isna()
    panel = panel.interpolate(limit=3, limit_area="inside")
    for lag in (168, 336, -168):
        panel = panel.fillna(panel.shift(lag))
    panel = panel.rename(columns=names)
    imputed = imputed.rename(columns=names)
    long = panel.stack().rename("y").reset_index()
    long.columns = ["ds", "unique_id", "y"]
    long["imputed"] = imputed.stack().to_numpy()
    return long[["unique_id", "ds", "y", "imputed"]].sort_values(["unique_id", "ds"], ignore_index=True)


def holiday_dates(years) -> pd.DataFrame:
    vic = holidays.country_holidays("AU", subdiv="VIC", years=years)
    return pd.DataFrame({"ds": pd.to_datetime(list(vic.keys())), "holiday": list(vic.values())})


def _days(values) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[D]").astype(np.int64)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ds = df["ds"]
    years = range(ds.dt.year.min() - 1, ds.dt.year.max() + 2)
    hol = np.unique(_days(holiday_dates(years)["ds"]))
    day = _days(ds.dt.normalize())
    nxt = np.searchsorted(hol, day)
    prv = np.searchsorted(hol, day, side="right") - 1
    xmas = day - _days(pd.to_datetime(ds.dt.year.astype(str) + "-12-25"))
    xmas = np.where(xmas < -180, xmas + 365, xmas)
    holiday = np.isin(day, hol)
    return df.assign(
        hour=ds.dt.hour,
        dow=ds.dt.day_of_week,
        doy=ds.dt.day_of_year,
        how=ds.dt.day_of_week * 24 + ds.dt.hour,
        holiday=holiday,
        offday=(ds.dt.day_of_week >= 5).to_numpy() | holiday,
        days_to_holiday=np.minimum(hol[nxt] - day, 3),
        days_since_holiday=np.minimum(day - hol[prv], 3),
        xmas=np.clip(xmas, -40, 40),
    )


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_parquet(RAW / "weather.parquet").drop_duplicates("ds")
    out = df.merge(weather, on="ds", how="left")
    out[["temp", "precip"]] = out.groupby("unique_id")[["temp", "precip"]].ffill().fillna(0)
    out["precip_3h"] = out.groupby("unique_id")["precip"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    return out


def build_panel(n: int = N_SENSORS) -> pd.DataFrame:
    counts = load_counts()
    panel = make_panel(counts, select_sensors(counts, n))
    return add_weather(add_calendar(panel))


BASE_FEATURES = ["sensor", "hour", "dow", "doy", "holiday", "offday", "temp", "precip"]
EXTRA_FEATURES = ["how", "days_to_holiday", "days_since_holiday", "xmas", "precip_3h"]


def lgb_features(df: pd.DataFrame, h: int, extended: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Target-derived features use lags >= h only, so one model covers all h steps without recursion.

    Returns the frame, all feature columns and the target-derived ones (NaN during warm-up).
    """
    uid = df["unique_id"]
    g = df.groupby("unique_id")["y"]
    lags = [lag for lag in (24, 48, 72, 168, 336, 504, 672) if lag >= h]
    feats = {f"lag_{lag}": g.shift(lag) for lag in lags}
    same_hour = pd.concat([g.shift(lag) for lag in (168, 336, 504, 672)], axis=1)
    feats["same_hour_4w"] = same_hour.mean(axis=1)
    shifted = g.shift(h)
    for w in (24, 168):
        feats[f"mean_{w}"] = shifted.groupby(uid).transform(lambda s, w=w: s.rolling(w).mean())
    if extended:
        feats["same_hour_4w_median"] = same_hour.median(axis=1)
        feats["same_hour_4w_std"] = same_hour.std(axis=1)
        feats["week_trend"] = feats["mean_168"] - feats["mean_168"].groupby(uid).shift(168)
    target_cols = list(feats)

    calendar = {}
    if extended:
        calendar["lag168_holiday"] = df.groupby("unique_id")["holiday"].shift(168).astype(float)
        if h < 168:
            calendar["lag24_offday"] = df.groupby("unique_id")["offday"].shift(24).astype(float)
    cols = BASE_FEATURES + (EXTRA_FEATURES if extended else []) + target_cols + list(calendar)
    return df.assign(**feats, **calendar), cols, target_cols
