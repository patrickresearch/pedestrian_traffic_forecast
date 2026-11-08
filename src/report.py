import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.backtest import RESULTS

MODELS = ["Chronos-Bolt", "LightGBM", "Prophet", "MSTL-ETS", "SeasonalNaive"]
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
    }
)


def leaderboard(metrics: pd.DataFrame) -> str:
    agg = metrics.groupby(["model", "horizon"])[["MASE", "sMAPE", "Coverage80"]].mean().reset_index()
    best = agg.groupby("horizon")["MASE"].transform("min") == agg["MASE"]
    agg["MASE"] = [f"**{v:.3f}**" if b else f"{v:.3f}" for v, b in zip(agg["MASE"], best)]
    agg["sMAPE"] = agg["sMAPE"].map("{:.1f} %".format)
    agg["Coverage80"] = agg["Coverage80"].map("{:.0f} %".format)
    wide = agg.pivot(index="model", columns="horizon").reindex(MODELS)
    lines = [
        "| Modell | MASE 24 h | MASE 7 d | sMAPE 24 h | sMAPE 7 d | Abdeckung 80 % (7 d) |",
        "|---|---|---|---|---|---|",
    ]
    for m, r in wide.iterrows():
        lines.append(
            f"| {m} | {r[('MASE', 24)]} | {r[('MASE', 168)]} | {r[('sMAPE', 24)]} "
            f"| {r[('sMAPE', 168)]} | {r[('Coverage80', 168)]} |"
        )
    return "\n".join(lines)


def holiday_split(fc: pd.DataFrame, history: pd.DataFrame) -> str:
    df = fc[(fc["horizon"] == 168) & ~fc["imputed"]].merge(history[["unique_id", "ds", "holiday"]])
    df["ase"] = (df["y"] - df["yhat"]).abs() / df["scale"]
    tab = df.groupby(["model", "holiday"], observed=True)["ase"].mean().unstack().reindex(MODELS)
    lines = ["| Modell | MASE normale Tage | MASE Feiertage |", "|---|---|---|"]
    lines += [f"| {m} | {r[False]:.3f} | {r[True]:.3f} |" for m, r in tab.iterrows()]
    return "\n".join(lines)


def per_sensor(metrics: pd.DataFrame) -> str:
    tab = metrics[metrics["horizon"] == 168].pivot_table(index="unique_id", columns="model", values="MASE")
    tab = tab[MODELS]
    lines = ["| Sensor | " + " | ".join(MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
    for s, r in tab.iterrows():
        cells = [f"**{v:.2f}**" if v == r.min() else f"{v:.2f}" for v in r]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def hero(fc: pd.DataFrame, sensor: str, origin: pd.Timestamp, path):
    week = fc[(fc["unique_id"] == sensor) & (fc["origin"] == origin) & (fc["horizon"] == 168)]
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 9), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        d = week[week["model"] == model].sort_values("ds")
        ax.fill_between(d["ds"], d["lo"], d["hi"], color=COLORS[model], alpha=0.2, linewidth=0)
        ax.plot(d["ds"], d["y"], color=INK, linewidth=1.2)
        ax.plot(d["ds"], d["yhat"], color=COLORS[model], linewidth=2)
        mase = (d["y"] - d["yhat"]).abs().mean() / d["scale"].iloc[0]
        ax.set_title(f"{model}  ·  MASE {mase:.2f}", loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=MUTED, alpha=0.2)
    axes[0].legend(["80 %-Intervall", "Ist", "Prognose"], loc="upper right", frameon=False, ncol=3, fontsize=9)
    fig.supylabel("Passanten pro Stunde", color=INK)
    fig.suptitle(f"{sensor}: 7-Tage-Prognose ab {origin + pd.Timedelta(hours=1):%d.%m.%Y}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(history: pd.DataFrame, sensor: str, path):
    d = history[history["unique_id"] == sensor]
    grid = d.groupby([d["ds"].dt.day_of_week, d["ds"].dt.hour])["y"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=BLUES)
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Stunde")
    ax.set_title(f"{sensor}: mittlere Passanten pro Stunde", loc="left", fontsize=11)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, pad=0.01).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    fc = pd.read_parquet(RESULTS / "forecasts.parquet")
    fc["model"] = fc["model"].astype(str)
    metrics = pd.read_parquet(RESULTS / "metrics.parquet")
    history = pd.read_parquet(RESULTS / "history.parquet")

    week = metrics[metrics["horizon"] == 168].groupby(["unique_id", "origin"])["MASE"].median()
    sensor, origin = (week - week.median()).abs().idxmin()
    hero(fc, sensor, origin, RESULTS / "hero.png")
    heatmap(history, sensor, RESULTS / "heatmap.png")

    report = "\n\n".join(
        [
            "## Leaderboard\n\n" + leaderboard(metrics),
            "## Feiertage\n\n" + holiday_split(fc, history),
            "## MASE pro Sensor (7 d)\n\n" + per_sensor(metrics),
        ]
    )
    (RESULTS / "leaderboard.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
