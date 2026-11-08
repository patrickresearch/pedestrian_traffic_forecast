from pathlib import Path

import holidays
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


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    days = holiday_dates(range(df["ds"].dt.year.min(), df["ds"].dt.year.max() + 1))["ds"]
    ds = df["ds"]
    return df.assign(
        hour=ds.dt.hour,
        dow=ds.dt.day_of_week,
        doy=ds.dt.day_of_year,
        holiday=ds.dt.normalize().isin(days),
        offday=(ds.dt.day_of_week >= 5) | ds.dt.normalize().isin(days),
    )


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_parquet(RAW / "weather.parquet").drop_duplicates("ds")
    out = df.merge(weather, on="ds", how="left")
    out[["temp", "precip"]] = out.groupby("unique_id")[["temp", "precip"]].ffill().fillna(0)
    return out


def build_panel(n: int = N_SENSORS) -> pd.DataFrame:
    counts = load_counts()
    panel = make_panel(counts, select_sensors(counts, n))
    return add_weather(add_calendar(panel))
