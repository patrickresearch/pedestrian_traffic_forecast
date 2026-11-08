"""Model order for tables and a fixed color per model, shared by report and app."""

MODELS = [
    "LightGBM-tuned",
    "LightGBM",
    "Chronos-2-Cov",
    "Chronos-2",
    "Chronos-Bolt",
    "Prophet",
    "MSTL-ETS",
    "SeasonalNaive",
]
COLORS = {
    "Chronos-Bolt": "#2a78d6",
    "LightGBM": "#eb6834",
    "Prophet": "#1baf7a",
    "MSTL-ETS": "#eda100",
    "SeasonalNaive": "#e87ba4",
    "LightGBM-tuned": "#008300",
    "Chronos-2": "#6250d6",
    "Chronos-2-Cov": "#e34948",
}
