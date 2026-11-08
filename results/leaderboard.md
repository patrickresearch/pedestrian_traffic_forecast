## Leaderboard

| Modell | MASE 24 h | MASE 7 d | sMAPE 24 h | sMAPE 7 d | Abdeckung 80 % (7 d) |
|---|---|---|---|---|---|
| Chronos-Bolt | 0.749 | 0.796 | 19.6 % | 21.5 % | 80 % |
| LightGBM | **0.711** | **0.769** | 16.6 % | 18.8 % | 69 % |
| Prophet | 1.098 | 1.019 | 23.3 % | 23.0 % | 84 % |
| MSTL-ETS | 1.233 | 1.303 | 23.0 % | 26.5 % | 96 % |
| SeasonalNaive | 1.016 | 1.012 | 24.5 % | 24.6 % | 87 % |

## Feiertage

| Modell | MASE normale Tage | MASE Feiertage |
|---|---|---|
| Chronos-Bolt | 0.768 | 1.572 |
| LightGBM | 0.745 | 1.420 |
| Prophet | 0.994 | 1.747 |
| MSTL-ETS | 1.253 | 2.646 |
| SeasonalNaive | 0.979 | 1.938 |

## MASE pro Sensor (7 d)

| Sensor | Chronos-Bolt | LightGBM | Prophet | MSTL-ETS | SeasonalNaive |
|---|---|---|---|---|---|
| Col620_T | 0.93 | **0.86** | 1.16 | 1.25 | 1.03 |
| ElFi_T | 0.76 | **0.72** | 0.93 | 1.28 | 0.99 |
| PriNW_T | 0.75 | **0.71** | 0.84 | 1.14 | 0.96 |
| QVN_T | 0.84 | **0.81** | 1.10 | 1.17 | 1.10 |
| SouthB_T | 0.81 | **0.71** | 1.13 | 1.45 | 1.03 |
| Swa123_T | 0.79 | **0.75** | 1.06 | 1.62 | 1.01 |
| Swa295_T | 0.84 | **0.83** | 1.06 | 1.12 | 1.13 |
| Swa31 | **0.65** | 0.76 | 0.88 | 1.39 | 0.86 |
