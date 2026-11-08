## Leaderboard

| Modell | MASE 24 h | MASE 7 d | sMAPE 24 h | sMAPE 7 d | Abdeckung 80 % (7 d) |
|---|---|---|---|---|---|
| LightGBM-tuned | **0.657** | **0.725** | 15.9 % | 17.9 % | 70 % |
| LightGBM | 0.711 | 0.769 | 16.6 % | 18.8 % | 69 % |
| Chronos-2-Cov | 0.728 | 0.736 | 20.8 % | 23.6 % | 75 % |
| Chronos-2 | 0.743 | 0.760 | 20.8 % | 23.1 % | 73 % |
| Chronos-Bolt | 0.749 | 0.796 | 19.6 % | 21.5 % | 80 % |
| Prophet | 1.098 | 1.019 | 23.3 % | 23.0 % | 84 % |
| MSTL-ETS | 1.233 | 1.303 | 23.0 % | 26.5 % | 96 % |
| SeasonalNaive | 1.016 | 1.012 | 24.5 % | 24.6 % | 87 % |

## Feiertage

| Modell | MASE normale Tage | MASE Feiertage |
|---|---|---|
| LightGBM-tuned | 0.700 | 1.405 |
| LightGBM | 0.745 | 1.420 |
| Chronos-2-Cov | 0.704 | 1.623 |
| Chronos-2 | 0.730 | 1.568 |
| Chronos-Bolt | 0.768 | 1.572 |
| Prophet | 0.994 | 1.747 |
| MSTL-ETS | 1.253 | 2.646 |
| SeasonalNaive | 0.979 | 1.938 |

## MASE pro Sensor (7 d)

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

## LightGBM: Tuning und Features (Validierung)

| Variante | MASE 24 h | MASE 7 d |
|---|---|---|
| v1 | 0.731 | 0.814 |
| v1 + Features | 0.725 | 0.838 |
| Tuning | 0.700 | 0.770 |
| Tuning + Features | 0.680 | 0.774 |
