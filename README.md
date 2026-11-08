# Pedestrian Traffic Forecast

Forecasting hourly pedestrian counts in Melbourne to answer one question:
**Is a time series foundation model worth it compared to a well-built LightGBM?**

![Hero](results/hero.png)

## Data

- [Melbourne Pedestrian Counting System](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/), hourly, ~100 sensors. The portal serves a rolling 2-year window, so the Covid years are excluded automatically.
- Weather: [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api) (temperature, precipitation)
- Holidays: `holidays` (AU, Victoria), including Melbourne Cup and AFL Grand Final Friday

Selection: the 8 sensors with ≥ 99 % coverage and the highest mean volume. Gaps are filled but not scored.

![Fingerprint](results/heatmap.png)

## Models

| Model | Idea |
|---|---|
| SeasonalNaive | value of the same hour one week earlier |
| MSTL-ETS | log scale, decomposition into daily and weekly seasonality, ETS on the remainder |
| Prophet | log scale, separate daily profiles for workdays and days off, holidays, weather |
| LightGBM | one global model across all sensors with calendar, weather and lags; one model per horizon using lags ≥ horizon, quantile models for the interval |
| LightGBM-tuned | like LightGBM, with Optuna parameters and additional features (see below) |
| Chronos-Bolt | `amazon/chronos-bolt-base` zero-shot, 2048 h context, no covariates |
| Chronos-2 | `amazon/chronos-2` zero-shot, 8192 h context, univariate |
| Chronos-2-Cov | like Chronos-2, plus holiday, day off, temperature and precipitation as known future values |

## Backtest

Rolling origin: 45 origins 8 days apart over the last year (so the weekday of the origin rotates), horizons 24 h and 7 days. Metrics: MASE (scale: in-sample seasonal naive error with lag 168), sMAPE, coverage of the 80 % interval.

| Model | MASE 24 h | MASE 7 d | sMAPE 24 h | sMAPE 7 d | 80 % coverage (7 d) |
|---|---|---|---|---|---|
| LightGBM-tuned | **0.657** | **0.725** | 15.9 % | 17.9 % | 70 % |
| LightGBM | 0.711 | 0.769 | 16.6 % | 18.8 % | 69 % |
| Chronos-2-Cov | 0.728 | 0.736 | 20.8 % | 23.6 % | 75 % |
| Chronos-2 | 0.743 | 0.760 | 20.8 % | 23.1 % | 73 % |
| Chronos-Bolt | 0.749 | 0.796 | 19.6 % | 21.5 % | 80 % |
| Prophet | 1.098 | 1.019 | 23.3 % | 23.0 % | 84 % |
| MSTL-ETS | 1.233 | 1.303 | 23.0 % | 26.5 % | 96 % |
| SeasonalNaive | 1.016 | 1.012 | 24.5 % | 24.6 % | 87 % |

MASE < 1 means better than the in-sample seasonal naive error. 360 windows per horizon (45 origins × 8 sensors).

### Holidays

| Model | MASE regular days | MASE holidays |
|---|---|---|
| LightGBM-tuned | 0.700 | 1.405 |
| LightGBM | 0.745 | 1.420 |
| Chronos-2-Cov | 0.704 | 1.623 |
| Chronos-2 | 0.730 | 1.568 |
| Chronos-Bolt | 0.768 | 1.572 |
| Prophet | 0.994 | 1.747 |
| MSTL-ETS | 1.253 | 2.646 |
| SeasonalNaive | 0.979 | 1.938 |

### MASE per sensor (7 d)

| Sensor | LightGBM-tuned | LightGBM | Chronos-2-Cov | Chronos-2 | Chronos-Bolt | Prophet | MSTL-ETS | SeasonalNaive |
|---|---|---|---|---|---|---|---|---|
| Col620_T | 0.82 | 0.86 | 0.76 | **0.75** | 0.93 | 1.16 | 1.25 | 1.03 |
| ElFi_T | **0.69** | 0.72 | 0.71 | 0.73 | 0.76 | 0.93 | 1.28 | 0.99 |
| PriNW_T | **0.66** | 0.71 | 0.69 | 0.74 | 0.75 | 0.84 | 1.14 | 0.96 |
| QVN_T | **0.76** | 0.81 | 0.80 | 0.82 | 0.84 | 1.10 | 1.17 | 1.10 |
| SouthB_T | **0.69** | 0.71 | 0.72 | 0.78 | 0.81 | 1.13 | 1.45 | 1.03 |
| Swa123_T | **0.72** | 0.75 | 0.77 | 0.78 | 0.79 | 1.06 | 1.62 | 1.01 |
| Swa295_T | **0.79** | 0.83 | 0.80 | 0.83 | 0.84 | 1.06 | 1.12 | 1.13 |
| Swa31 | 0.69 | 0.76 | **0.64** | 0.64 | 0.65 | 0.88 | 1.39 | 0.86 |

## Optimizing LightGBM

Optuna (TPE, 50 trials per horizon) on a 10-week validation window directly before the first backtest origin, trained on everything before it. Since all lags are ≥ horizon, each validation row is a genuine h-step-ahead forecast. The number of rounds comes from early stopping. The search covered learning rate, tree size, regularization, sampling, loss (L2, L1, Huber) and target (log(y) or deviation from the 4-week same-hour mean).

New features: hour of week, days to/since holiday (capped at 3), distance to Christmas, median and standard deviation of the same hour over 4 weeks, weekly trend, holiday one week earlier, day off on the previous day, precipitation over the last 3 hours.

| Variant | MASE 24 h | MASE 7 d |
|---|---|---|
| v1 | 0.731 | 0.814 |
| v1 + Features | 0.725 | 0.838 |
| Tuning | 0.700 | 0.770 |
| Tuning + Features | 0.680 | 0.774 |

Both horizons pick L1 loss on log(y), small trees (26–27 leaves) and only 30–45 % of the features per tree.

## Findings

**In short: the tuned LightGBM wins, Chronos-2 with covariates gets within 2 % without any training.**

- **Optimization:** LightGBM-tuned improves MASE over v1 by 8 % (24 h, 0.711 → 0.657) and 6 % (7 d, 0.769 → 0.725) and wins about two thirds of the windows against v1. Most of the gain comes from tuning. Both horizons pick L1 loss, which directly optimizes the absolute error that MASE is based on. The new features help the 24 h model and are neutral for the 7-day model. The most important new feature is the 4-week same-hour median (about 30 % of the gain), which is more robust to single holidays than the mean.
- **Chronos-2 vs. Chronos-Bolt:** Chronos-2 is 5 % better at 7 days (0.760 vs. 0.796) and on par at 24 h. The longer context and the direct 168-step forecast pay off at the long horizon.
- **Covariates:** Holidays and weather improve Chronos-2 by 2–3 % (7 d: 0.736 vs. 0.760). That puts Chronos-2-Cov almost level with LightGBM-tuned at 7 days (0.736 vs. 0.725), and it even wins slightly more individual windows (185 vs. 175). At 24 h LightGBM stays clearly ahead (0.657 vs. 0.728) because it uses the lags of the last few days.
- **Holidays:** All models roughly double their error. LightGBM-tuned stays best there (1.41). Chronos-2-Cov is almost as good on regular days (0.704 vs. 0.700) but worse on holidays (1.62), and no better than without covariates. A holiday flag alone does not tell the foundation model how strong the effect of a holiday is.
- **Intervals:** Chronos-Bolt is best calibrated (80–84 % coverage for the 80 % interval), Chronos-2 reaches 73–77 %. The LightGBM quantile models are too narrow (68–70 %), MSTL-ETS is far too wide (96 %).
- **Effort:** None of the Chronos variants need features or training: 1–3 minutes for 45 origins on MPS. LightGBM-tuned additionally needs about 20 minutes of tuning and several times the compute in the backtest.
- **Surprise:** MSTL-ETS and Prophet are no better than last week's value. For stable commuter locations the seasonal naive approach is already strong, and LightGBM owes about three quarters of its gain to the values of the same hour over the last four weeks.

For digital signage this means: Chronos-2 with calendar and weather is a very good baseline that can be rolled out to new locations without data science effort, and for weekly forecasts it is barely worse than a tuned LightGBM. For daily forecasts and holidays, a dedicated model pays off.

Caveat: Prophet, LightGBM and Chronos-2-Cov receive the actually observed weather as a regressor (oracle). With weather forecasts they would be slightly worse.

## Structure

```
data/download.py        download counts and weather to data/raw/
src/features.py         sensor selection, hourly grid, calendar, weather
src/models.py           all models behind one interface
src/metrics.py          MASE, sMAPE, coverage
src/tune.py             Optuna tuning for LightGBM on a window before the backtest
src/backtest.py         rolling backtest, writes results/*.parquet
src/report.py           README figures and tables
notebooks/              01_eda, 02_baselines, 03_foundation
app/streamlit_app.py    loads precomputed results only
```

## Reproduce

```bash
brew install libomp      # for LightGBM on macOS
uv sync
uv run python data/download.py
uv run python -m src.tune         # ~20 min, writes results/lgb_params.json
uv run python -m src.backtest     # --debug for details, --resume after an interruption
uv run python -m src.report
uv run streamlit run app/streamlit_app.py
```

PyTorch and LightGBM load different OpenMP runtimes that deadlock each other in one process, so Chronos runs in a separate process.

## Deployment

Streamlit Community Cloud, entry point `app/streamlit_app.py`. The app only needs `app/requirements.txt` and `results/`; Chronos is not run inside the app.
