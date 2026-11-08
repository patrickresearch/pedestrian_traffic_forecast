import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import coverage, mase, smape
from src.palette import COLORS, MODELS

RESULTS = ROOT / "results"
DEFAULT = ["LightGBM-tuned", "Chronos-2-Cov", "Chronos-Bolt", "SeasonalNaive"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HORIZONS = {"24 hours": 24, "7 days": 168}
GITHUB = "https://github.com/patrickresearch/pedestrian_traffic_forecast"
WANDB = "https://wandb.ai/kahala_patrick-stanford-university/pedestrian-forecast"
FINDINGS = """
- **Tuning pays off.** Optuna tuning and new features improve LightGBM by 8 % (24 h) and 6 % (7 d) over v1.
  Both horizons pick L1 loss, which directly optimizes the absolute error behind MASE.
- **Chronos-2 beats Chronos-Bolt** by 5 % at 7 days. Longer context and a direct 168-step forecast help at the long horizon.
- **Covariates close the gap.** With holidays and weather, Chronos-2 gets within 2 % of the tuned LightGBM at 7 days
  and wins slightly more individual windows, without any training.
- **Short horizons favour LightGBM**, because the 24 h model uses the lags of the last few days.
- **Holidays double every model's error.** LightGBM-tuned stays best there, a holiday flag alone does not tell Chronos how strong the effect is.
- **Intervals:** Chronos-Bolt is best calibrated, the LightGBM quantile models are too narrow, MSTL-ETS far too wide.

*Caveat: models with weather covariates receive observed weather (oracle), so their errors are slightly optimistic.*
"""

st.set_page_config(page_title="Melbourne Pedestrian Forecast", layout="wide")


@st.cache_data
def load():
    fc = pd.read_parquet(RESULTS / "forecasts.parquet")
    fc[["model", "unique_id"]] = fc[["model", "unique_id"]].astype(str)
    return fc, pd.read_parquet(RESULTS / "history.parquet"), pd.read_parquet(RESULTS / "metrics.parquet")


@st.cache_data
def error_profiles(fc: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = fc[(fc["horizon"] == 168) & ~fc["imputed"]].merge(history[["unique_id", "ds", "holiday"]])
    d["ase"] = (d["y"] - d["yhat"]).abs() / d["scale"]
    day = ((d["ds"] - d["origin"]) / pd.Timedelta(hours=1) - 1) // 24 + 1
    return d.groupby(["model", day])["ase"].mean().unstack(0), d.groupby(["model", "holiday"])["ase"].mean().unstack()


fc, history, metrics = load()
board = metrics.groupby(["horizon", "model"])["MASE"].mean().unstack(0)

with st.sidebar:
    st.header("Selection")
    sensor = st.selectbox("Sensor", sorted(fc["unique_id"].unique()))
    available = [m for m in MODELS if m in set(fc["model"])]
    models = st.multiselect("Models", available, default=[m for m in DEFAULT if m in available] or available)
    horizon = HORIZONS[st.radio("Horizon", list(HORIZONS), index=1, horizontal=True)]
    band = st.selectbox("80 % interval for", ["none", *models], index=1 if models else 0)
    origins = sorted(fc["origin"].unique())
    start, end = st.select_slider(
        "Period (forecast origin)",
        options=origins,
        value=(origins[max(0, len(origins) - 6)], origins[-1]),
        format_func=lambda t: f"{pd.Timestamp(t) + pd.Timedelta(hours=1):%Y-%m-%d}",
    )
    st.divider()
    st.markdown(f"[Code on GitHub]({GITHUB})  \n[Experiments on W&B]({WANDB})")

st.title("Melbourne Pedestrian Forecast")
st.caption(
    "Is a time series foundation model worth it compared to a well-built LightGBM? "
    "Rolling backtest on hourly counts of the Melbourne Pedestrian Counting System: "
    f"{metrics['origin'].nunique()} origins × {metrics['unique_id'].nunique()} sensors. All forecasts are precomputed."
)

week = board[168].sort_values()
best = week.index[0]
k1, k2, k3 = st.columns(3)
k1.metric("Best model (7 d MASE)", best, f"{week.iloc[0]:.3f}", delta_color="off")
if "Chronos-2-Cov" in week and best != "Chronos-2-Cov":
    gap = week["Chronos-2-Cov"] / week.iloc[0] - 1
    k2.metric("Chronos-2-Cov, zero-shot", f"{week['Chronos-2-Cov']:.3f}", f"+{gap:.1%} vs. {best}", delta_color="off")
if "SeasonalNaive" in week:
    gain = 1 - week.iloc[0] / week["SeasonalNaive"]
    k3.metric("Gain over last week's value", f"{gain:.0%}", f"SeasonalNaive {week['SeasonalNaive']:.3f}", delta_color="off")

sel = fc[
    (fc["unique_id"] == sensor)
    & (fc["horizon"] == horizon)
    & fc["model"].isin(models)
    & fc["origin"].between(start, end)
]

if sel.empty:
    st.info("Select at least one model.")
    st.stop()

scored = sel[~sel["imputed"]]
rows = []
for model, g in scored.groupby("model"):
    windows = g.groupby("origin")
    rows.append(
        {
            "Model": model,
            "MASE": windows.apply(lambda w: mase(w["y"], w["yhat"], w["scale"].iloc[0])).mean(),
            "sMAPE %": windows.apply(lambda w: smape(w["y"].to_numpy(), w["yhat"].to_numpy())).mean(),
            "80 % coverage": coverage(g["y"], g["lo"], g["hi"]),
        }
    )
table = pd.DataFrame(rows).set_index("Model").reindex([m for m in MODELS if m in models]).sort_values("MASE")

col_plot, col_table = st.columns([3, 1])
with col_table:
    st.subheader("Error in period")
    st.dataframe(table.style.format("{:.2f}").highlight_min("MASE", color="#cde2fb"), width="stretch")
    st.caption(f"{sel['origin'].nunique()} forecast windows, {sensor}")

with col_plot:
    grid = pd.date_range(sel["ds"].min(), sel["ds"].max(), freq="h")
    actual = sel.drop_duplicates("ds").set_index("ds")["y"].reindex(grid)
    fig = go.Figure()
    if band != "none":
        b = sel[sel["model"] == band].set_index("ds").reindex(grid)
        fig.add_trace(go.Scatter(x=grid, y=b["hi"], line_width=0, showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=grid, y=b["lo"], fill="tonexty", line_width=0, fillcolor=COLORS[band],
                opacity=0.2, name=f"{band} 80 %", hoverinfo="skip",
            )
        )
    fig.add_trace(go.Scatter(x=grid, y=actual, name="Actual", line=dict(color="#0b0b0b", width=1.5)))
    for model in table.index:
        m = sel[sel["model"] == model].set_index("ds")["yhat"].reindex(grid)
        fig.add_trace(go.Scatter(x=grid, y=m, name=model, line=dict(color=COLORS[model], width=2)))
    fig.update_layout(
        height=480,
        hovermode="x unified",
        yaxis_title="Pedestrians per hour",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")

tab_findings, tab_heat, tab_board = st.tabs(["Findings", "Fingerprint", "Leaderboard"])

with tab_findings:
    st.markdown(FINDINGS)
    by_day, by_holiday = error_profiles(fc, history)
    shown = [m for m in MODELS if m in models]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for model in shown:
            fig.add_trace(
                go.Scatter(
                    x=by_day.index, y=by_day[model], name=model, mode="lines+markers",
                    line=dict(color=COLORS[model], width=2),
                )
            )
        fig.update_layout(
            title="MASE by day in the 7-day window", xaxis_title="Day after origin", yaxis_title="MASE",
            height=380, margin=dict(t=50, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        h = by_holiday.reindex(shown)
        fig = go.Figure(
            [
                go.Bar(x=h.index, y=h[False], name="Regular days", marker_color="#6da7ec"),
                go.Bar(x=h.index, y=h[True], name="Holidays", marker_color="#0d366b"),
            ]
        )
        fig.update_layout(
            title="MASE on regular days vs. holidays (7 d)", yaxis_title="MASE", barmode="group",
            height=380, margin=dict(t=50, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, width="stretch")
    st.caption("Both charts follow the model selection in the sidebar and cover all sensors and origins.")

with tab_heat:
    h = history[history["unique_id"] == sensor]
    heat = h.groupby([h["ds"].dt.day_of_week, h["ds"].dt.hour])["y"].mean().unstack()
    fig = go.Figure(
        go.Heatmap(
            z=heat.to_numpy(), x=list(range(24)), y=DAYS,
            colorscale=[[0, "#cde2fb"], [0.33, "#6da7ec"], [0.66, "#256abf"], [1, "#0d366b"]],
            hovertemplate="%{y} %{x}:00<br>%{z:.0f} pedestrians<extra></extra>",
        )
    )
    fig.update_layout(height=320, xaxis_title="Hour", yaxis_autorange="reversed", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Mean pedestrians per hour, {sensor}")

with tab_board:
    board_md = RESULTS / "leaderboard.md"
    if board_md.exists():
        st.markdown(board_md.read_text())
