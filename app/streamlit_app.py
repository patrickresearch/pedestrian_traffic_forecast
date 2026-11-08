import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import coverage, mase, smape

RESULTS = ROOT / "results"
MODELS = ["Chronos-Bolt", "LightGBM", "Prophet", "MSTL-ETS", "SeasonalNaive"]
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
DAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
HORIZONS = {"24 Stunden": 24, "7 Tage": 168}

st.set_page_config(page_title="Passantenprognose Melbourne", layout="wide")


@st.cache_data
def load():
    fc = pd.read_parquet(RESULTS / "forecasts.parquet")
    fc[["model", "unique_id"]] = fc[["model", "unique_id"]].astype(str)
    return fc, pd.read_parquet(RESULTS / "history.parquet")


fc, history = load()

with st.sidebar:
    st.header("Auswahl")
    sensor = st.selectbox("Sensor", sorted(fc["unique_id"].unique()))
    models = st.multiselect("Modelle", MODELS, default=MODELS)
    horizon = HORIZONS[st.radio("Horizont", list(HORIZONS), index=1, horizontal=True)]
    band = st.selectbox("80 %-Intervall für", ["keins", *models], index=1 if models else 0)
    origins = sorted(fc["origin"].unique())
    start, end = st.select_slider(
        "Zeitraum (Prognoseursprung)",
        options=origins,
        value=(origins[max(0, len(origins) - 6)], origins[-1]),
        format_func=lambda t: f"{pd.Timestamp(t) + pd.Timedelta(hours=1):%d.%m.%Y}",
    )

st.title("Passantenprognose Melbourne")
st.caption(
    "Rollierender Backtest auf stündlichen Zählungen des Melbourne Pedestrian Counting System. "
    "Alle Prognosen sind vorberechnet."
)

sel = fc[
    (fc["unique_id"] == sensor)
    & (fc["horizon"] == horizon)
    & fc["model"].isin(models)
    & fc["origin"].between(start, end)
]

if sel.empty:
    st.info("Mindestens ein Modell auswählen.")
    st.stop()

scored = sel[~sel["imputed"]]
rows = []
for model, g in scored.groupby("model"):
    windows = g.groupby("origin")
    rows.append(
        {
            "Modell": model,
            "MASE": windows.apply(lambda w: mase(w["y"], w["yhat"], w["scale"].iloc[0])).mean(),
            "sMAPE %": windows.apply(lambda w: smape(w["y"].to_numpy(), w["yhat"].to_numpy())).mean(),
            "Abdeckung 80 %": coverage(g["y"], g["lo"], g["hi"]),
        }
    )
table = pd.DataFrame(rows).set_index("Modell").reindex([m for m in MODELS if m in models]).sort_values("MASE")

col_plot, col_table = st.columns([3, 1])
with col_table:
    st.subheader("Fehler im Zeitraum")
    st.dataframe(table.style.format("{:.2f}").highlight_min("MASE", color="#cde2fb"), width="stretch")
    st.caption(f"{sel['origin'].nunique()} Prognosefenster")

with col_plot:
    grid = pd.date_range(sel["ds"].min(), sel["ds"].max(), freq="h")
    actual = sel.drop_duplicates("ds").set_index("ds")["y"].reindex(grid)
    fig = go.Figure()
    if band != "keins":
        b = sel[sel["model"] == band].set_index("ds").reindex(grid)
        fig.add_trace(go.Scatter(x=grid, y=b["hi"], line_width=0, showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=grid, y=b["lo"], fill="tonexty", line_width=0, fillcolor=COLORS[band],
                opacity=0.2, name=f"{band} 80 %", hoverinfo="skip",
            )
        )
    fig.add_trace(go.Scatter(x=grid, y=actual, name="Ist", line=dict(color="#0b0b0b", width=1.5)))
    for model in table.index:
        m = sel[sel["model"] == model].set_index("ds")["yhat"].reindex(grid)
        fig.add_trace(go.Scatter(x=grid, y=m, name=model, line=dict(color=COLORS[model], width=2)))
    fig.update_layout(
        height=480,
        hovermode="x unified",
        yaxis_title="Passanten pro Stunde",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")

tab_heat, tab_board = st.tabs(["Fingerabdruck", "Gesamt-Leaderboard"])

with tab_heat:
    h = history[history["unique_id"] == sensor]
    heat = h.groupby([h["ds"].dt.day_of_week, h["ds"].dt.hour])["y"].mean().unstack()
    fig = go.Figure(
        go.Heatmap(
            z=heat.to_numpy(), x=list(range(24)), y=DAYS,
            colorscale=[[0, "#cde2fb"], [0.33, "#6da7ec"], [0.66, "#256abf"], [1, "#0d366b"]],
            hovertemplate="%{y} %{x}:00 Uhr<br>%{z:.0f} Passanten<extra></extra>",
        )
    )
    fig.update_layout(height=320, xaxis_title="Stunde", yaxis_autorange="reversed", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with tab_board:
    board = RESULTS / "leaderboard.md"
    if board.exists():
        st.markdown(board.read_text())
