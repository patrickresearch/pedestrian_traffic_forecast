# Pedestrian Traffic Forecast

Stündliche Passantenzahlen in Melbourne vorhersagen und dabei eine Frage beantworten:
**Lohnt sich ein Zeitreihen-Foundation-Model gegenüber einem gut gebauten LightGBM?**

![Hero](results/hero.png)

## Daten

- [Melbourne Pedestrian Counting System](https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-monthly-counts-per-hour/), stündlich, ~100 Sensoren. Das Portal liefert ein rollierendes 2-Jahres-Fenster, Corona-Jahre sind damit automatisch ausgeschlossen.
- Wetter: [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api) (Temperatur, Niederschlag)
- Feiertage: `holidays` (AU, Victoria), inkl. Melbourne Cup und AFL Grand Final Friday

Auswahl: 8 Sensoren mit ≥ 99 % Abdeckung und dem höchsten mittleren Aufkommen. Lücken werden aufgefüllt, aber nicht bewertet.

![Fingerabdruck](results/heatmap.png)

## Modelle

| Modell | Idee |
|---|---|
| SeasonalNaive | Wert derselben Stunde der Vorwoche |
| MSTL-ETS | log-Skala, Zerlegung in Tages- und Wochensaison, ETS auf dem Rest |
| Prophet | log-Skala, getrennte Tagesprofile für Werk- und Freitage, Feiertage, Wetter |
| LightGBM | ein globales Modell über alle Sensoren, Kalender, Wetter, Lags; je Horizont ein Modell mit Lags ≥ Horizont, Quantilmodelle für das Intervall |
| LightGBM-tuned | wie LightGBM, mit Optuna-Parametern und zusätzlichen Features (s. u.) |
| Chronos-Bolt | `amazon/chronos-bolt-base` zero-shot, 2048 h Kontext, keine Kovariaten |
| Chronos-2 | `amazon/chronos-2` zero-shot, 8192 h Kontext, univariat |
| Chronos-2-Cov | wie Chronos-2, dazu Feiertag, Frei-Tag, Temperatur und Niederschlag als bekannte Zukunftswerte |

## Backtest

Rollierender Ursprung: 45 Ursprünge im Abstand von 8 Tagen über das letzte Jahr (dadurch wechselt der Wochentag des Ursprungs), Horizont 24 h und 7 Tage. Metriken: MASE (Skala: saisonaler Naive-Fehler mit Lag 168 im Training), sMAPE, Abdeckung des 80 %-Intervalls.

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

MASE < 1 heißt: besser als der saisonale Naive-Fehler im Training. 360 Fenster je Horizont (45 Ursprünge × 8 Sensoren).

### Feiertage

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

### MASE pro Sensor (7 d)

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

## LightGBM optimieren

Optuna (TPE, 50 Trials je Horizont) auf einem Validierungsfenster von 10 Wochen direkt vor dem ersten Backtest-Ursprung, Training davor. Da alle Lags ≥ Horizont sind, entspricht jede Validierungszeile einer echten h-Schritt-Prognose. Die Rundenzahl kommt aus Early Stopping. Gesucht wurde über Lernrate, Baumgröße, Regularisierung, Sampling, Verlustfunktion (L2, L1, Huber) und Zielgröße (log(y) oder Abweichung vom 4-Wochen-Mittel).

Neue Features: Stunde der Woche, Tage bis/seit Feiertag (gekappt bei 3), Abstand zu Weihnachten, Median und Streuung derselben Stunde über 4 Wochen, Wochentrend, Feiertag vor einer Woche, Frei-Tag am Vortag, Niederschlag der letzten 3 Stunden.

| Variante | MASE 24 h | MASE 7 d |
|---|---|---|
| v1 | 0.731 | 0.814 |
| v1 + Features | 0.725 | 0.838 |
| Tuning | 0.700 | 0.770 |
| Tuning + Features | 0.680 | 0.774 |

Beide Horizonte wählen L1-Verlust auf log(y), kleine Bäume (26–27 Blätter) und nur 30–45 % der Features pro Baum.

## Erkenntnisse

**Kurz: Der getunte LightGBM gewinnt, Chronos-2 mit Kovariaten kommt ohne Training bis auf 2 % heran.**

- **Optimierung:** LightGBM-tuned verbessert MASE gegenüber v1 um 8 % (24 h, 0.711 → 0.657) und 6 % (7 d, 0.769 → 0.725) und gewinnt rund zwei Drittel der Fenster gegen v1. Den größten Teil bringt das Tuning. Beide Horizonte wählen L1-Verlust, der direkt den absoluten Fehler optimiert, auf dem MASE beruht. Die neuen Features helfen beim 24-h-Modell, beim 7-Tage-Modell sind sie neutral. Wichtigstes neues Feature ist der Median derselben Stunde über 4 Wochen (rund 30 % des Gains), er ist robuster gegen einzelne Feiertage als der Mittelwert.
- **Chronos-2 gegen Chronos-Bolt:** Chronos-2 ist bei 7 Tagen 5 % besser (0.760 gegen 0.796), bei 24 h gleichauf. Der längere Kontext und die direkte Prognose über 168 Schritte zahlen sich beim langen Horizont aus.
- **Kovariaten:** Feiertage und Wetter verbessern Chronos-2 um 2–3 % (7 d: 0.736 gegen 0.760). Damit ist Chronos-2-Cov bei 7 Tagen fast gleichauf mit LightGBM-tuned (0.736 gegen 0.725) und gewinnt sogar etwas mehr Einzelfenster (185 gegen 175). Bei 24 h bleibt LightGBM klar vorn (0.657 gegen 0.728), weil es die Lags der letzten Tage nutzt.
- **Feiertage:** Alle Modelle verdoppeln ihren Fehler. LightGBM-tuned bleibt dort am besten (1.41). Chronos-2-Cov ist an normalen Tagen fast so gut (0.704 gegen 0.700), an Feiertagen aber schlechter (1.62) und nicht besser als ohne Kovariaten. Das Feiertags-Flag allein erklärt dem Foundation Model nicht, wie stark ein Feiertag wirkt.
- **Intervalle:** Chronos-Bolt ist am besten kalibriert (80–84 % Abdeckung beim 80 %-Intervall), Chronos-2 liegt bei 73–77 %. Die LightGBM-Quantilmodelle sind zu schmal (68–70 %), MSTL-ETS ist viel zu breit (96 %).
- **Aufwand:** Alle Chronos-Varianten brauchen keine Features und kein Training: 1–3 Minuten für 45 Ursprünge auf MPS. LightGBM-tuned braucht dafür zusätzlich rund 20 Minuten Tuning und ein Mehrfaches an Rechenzeit im Backtest.
- **Überraschung:** MSTL-ETS und Prophet sind nicht besser als die Vorwoche. Für stabile Pendlerstandorte ist der saisonale Naive-Ansatz schon stark, und LightGBM verdankt rund drei Viertel seines Gains den Werten derselben Stunde in den letzten vier Wochen.

Für Digital Signage heißt das: Chronos-2 mit Kalender und Wetter ist eine sehr gute Ausgangsbasis, die man ohne Data-Science-Aufwand für neue Standorte nutzen kann, und für Wochenprognosen kaum schlechter als ein getunter LightGBM. Für Tagesprognosen und Feiertage lohnt sich das eigene Modell.

Einschränkung: Prophet, LightGBM und Chronos-2-Cov bekommen das tatsächlich beobachtete Wetter als Regressor (Oracle). Mit Wetterprognosen wären sie etwas schlechter.

## Struktur

```
data/download.py        Zählungen und Wetter nach data/raw/ laden
src/features.py         Sensorauswahl, Stundenraster, Kalender, Wetter
src/models.py           alle Modelle mit einheitlicher Schnittstelle
src/metrics.py          MASE, sMAPE, Abdeckung
src/tune.py             Optuna-Tuning für LightGBM auf einem Zeitraum vor dem Backtest
src/backtest.py         rollierender Backtest, schreibt results/*.parquet
src/report.py           README-Grafiken und Tabellen
notebooks/              01_eda, 02_baselines, 03_foundation
app/streamlit_app.py    lädt nur vorberechnete Ergebnisse
```

## Reproduzieren

```bash
brew install libomp      # für LightGBM auf macOS
uv sync
uv run python data/download.py
uv run python -m src.tune         # ~20 min, schreibt results/lgb_params.json
uv run python -m src.backtest     # --debug für Details, --resume nach Abbruch
uv run python -m src.report
uv run streamlit run app/streamlit_app.py
```

PyTorch und LightGBM laden unterschiedliche OpenMP-Laufzeiten, die sich in einem Prozess gegenseitig blockieren. Chronos läuft deshalb in einem eigenen Prozess.

## Deployment

Streamlit Community Cloud, Einstiegspunkt `app/streamlit_app.py`. Die App braucht nur `app/requirements.txt` und `results/`, Chronos wird nicht in der App gerechnet.
