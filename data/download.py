from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).parent / "raw"
COUNTS_URL = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
    "pedestrian-counting-system-monthly-counts-per-hour/exports/parquet"
)
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
MELBOURNE = {"latitude": -37.8136, "longitude": 144.9631}


def download_counts() -> pd.DataFrame:
    path = RAW / "counts.parquet"
    with requests.get(COUNTS_URL, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return pd.read_parquet(path)


def download_weather(start: str, end: str) -> pd.DataFrame:
    params = {
        **MELBOURNE,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,precipitation",
        "timezone": "Australia/Melbourne",
    }
    r = requests.get(WEATHER_URL, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(hourly["time"]),
            "temp": hourly["temperature_2m"],
            "precip": hourly["precipitation"],
        }
    )
    df.to_parquet(RAW / "weather.parquet", index=False)
    return df


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    counts = download_counts()
    dates = pd.to_datetime(counts["sensing_date"])
    weather = download_weather(f"{dates.min():%Y-%m-%d}", f"{dates.max():%Y-%m-%d}")
    print(f"counts: {len(counts):,} rows, {counts['sensor_name'].nunique()} sensors")
    print(f"weather: {len(weather):,} rows")
