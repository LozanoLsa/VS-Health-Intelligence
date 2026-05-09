"""
Trend and forecast visualizations for Monthly Trends and Predictive Simulation tabs.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from src.ml.forecasting import MachineForecast, _forecast_series
from src.utils.logger import get_logger

log = get_logger(__name__)

NAVY   = "#0D1B2A"
WHITE  = "#FFFFFF"
VSM_COLORS = {"Alpha": "#2ECC71", "Beta": "#F39C12", "Gamma": "#E74C3C"}
HEALTH_COLORS = {"Healthy": "#2ECC71", "Monitor": "#F39C12", "Critical": "#E74C3C"}
FORECAST_COLOR = "#E84C1F"


def _month_labels(months: list[str], max_labels: int = 10) -> list[str]:
    n    = len(months)
    step = max(1, n // max_labels)
    return [m if i % step == 0 else "" for i, m in enumerate(months)]


def plot_vsm_monthly_trend(monthly_df: pd.DataFrame,
                            metric_col: str = "health_score",
                            vsm_filter: str = "All") -> plt.Figure:
    """Line chart: selected metric over time, one line per VSM."""
    if vsm_filter != "All":
        monthly_df = monthly_df[monthly_df["vsm"] == vsm_filter]

    vsm_agg = (
        monthly_df.groupby(["year_month", "vsm"])[metric_col]
        .mean().reset_index()
        .sort_values("year_month")
    )
    months = sorted(vsm_agg["year_month"].unique())

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor("#F9F9F9")

    for vsm in ["Alpha", "Beta", "Gamma"]:
        data = vsm_agg[vsm_agg["vsm"] == vsm].sort_values("year_month")
        if data.empty:
            continue
        xs = [months.index(m) for m in data["year_month"]]
        ax.plot(xs, data[metric_col], color=VSM_COLORS[vsm],
                linewidth=2.5, label=f"VSM {vsm}",
                marker="o", markersize=5, zorder=3)

    # Health zone bands (only for health_score)
    if metric_col == "health_score":
        ax.axhspan(70, 100, alpha=0.06, color="#2ECC71", zorder=0)
        ax.axhspan(40,  70, alpha=0.06, color="#F39C12", zorder=0)
        ax.axhspan( 0,  40, alpha=0.06, color="#E74C3C", zorder=0)
        ax.axhline(70, color="#2ECC71", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(40, color="#E74C3C", lw=0.8, ls="--", alpha=0.5)

    labels = _month_labels(months)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"Monthly {metric_col.replace('_',' ').title()} by VSM", fontsize=10, color=NAVY)
    fig.tight_layout(pad=0.8)
    return fig


def plot_monthly_failures_bar(monthly_df: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart of monthly failures by area."""
    agg = (
        monthly_df.groupby(["year_month", "area"])["failures_count"]
        .sum().reset_index().sort_values("year_month")
    )
    months = sorted(agg["year_month"].unique())
    areas  = ["Machining", "Painting", "Assembly"]
    colors = {"Machining": "#1A5276", "Painting": "#2980B9", "Assembly": "#85C1E9"}

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor("#F9F9F9")

    bottoms = np.zeros(len(months))
    for area in areas:
        vals = []
        for m in months:
            row = agg[(agg["year_month"] == m) & (agg["area"] == area)]
            vals.append(float(row["failures_count"].sum()) if not row.empty else 0.0)
        bars = ax.bar(range(len(months)), vals, bottom=bottoms,
                      color=colors[area], label=area, alpha=0.85, width=0.7)
        # Value labels inside each segment (only if segment is tall enough)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[i] + val / 2,
                    str(int(val)),
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                )
        bottoms += np.array(vals)

    labels = _month_labels(months)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Failures", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.set_title("Monthly Failure Count by Area", fontsize=10, color=NAVY)
    fig.tight_layout(pad=0.8)
    return fig


def plot_health_calendar(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a styled pivot: machines as rows, months as columns,
    health_score as values.  Caller applies st.dataframe() with gradient.
    """
    pivot = monthly_df.pivot_table(
        index="machine_id", columns="year_month",
        values="health_score", aggfunc="mean"
    ).round(1)
    return pivot


def plot_machine_forecast(monthly_df: pd.DataFrame,
                           machine_id: str,
                           horizon: int = 1) -> plt.Figure:
    """
    Actual historical health score + fitted trend + forecast band.
    """
    grp    = monthly_df[monthly_df["machine_id"] == machine_id].sort_values("year_month")
    months = list(grp["year_month"].values)
    values = grp["health_score"].values.astype(float)
    n      = len(values)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor("#F9F9F9")

    # Health zone bands
    ax.axhspan(70, 105, alpha=0.07, color="#2ECC71", zorder=0)
    ax.axhspan(40,  70, alpha=0.07, color="#F39C12", zorder=0)
    ax.axhspan( 0,  40, alpha=0.07, color="#E74C3C", zorder=0)
    ax.axhline(70, color="#2ECC71", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(40, color="#E74C3C", lw=0.8, ls="--", alpha=0.6)

    # Historical line
    ax.plot(range(n), values, color=NAVY, linewidth=2.2,
            marker="o", markersize=6, label="Actual", zorder=4)

    # Trend line
    if n >= 3:
        x       = np.arange(n)
        coeffs  = np.polyfit(x, values, 1)
        trend   = np.polyval(coeffs, x)
        ax.plot(range(n), trend, color="#555", linewidth=1.2,
                linestyle="--", alpha=0.6, label="Trend", zorder=3)

    # Forecast
    fx, fy, flo, fhi = [], [], [], []
    for h in range(1, horizon + 1):
        fc, lo, hi = _forecast_series(values, h)
        fx.append(n - 1 + h)
        fy.append(float(np.clip(fc,  0, 100)))
        flo.append(float(np.clip(lo, 0, 100)))
        fhi.append(float(np.clip(hi, 0, 100)))

    # Connect last actual to first forecast
    ax.plot([n - 1] + fx, [values[-1]] + fy,
            color=FORECAST_COLOR, linewidth=2.2, linestyle="--",
            marker="D", markersize=7, label=f"Forecast (+{horizon}mo)", zorder=5)
    ax.fill_between([n - 1] + fx, [values[-1]] + flo, [values[-1]] + fhi,
                    alpha=0.18, color=FORECAST_COLOR, label="80% CI")

    # "Today" divider
    ax.axvline(x=n - 0.5, color="#888", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(n - 0.45, 5, "Forecast", fontsize=8, color="#888")

    # X axis labels
    x_labels = months + [f"+{h}mo" for h in range(1, horizon + 1)]
    step = max(1, len(x_labels) // 10)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(
        [x_labels[i] if i % step == 0 else "" for i in range(len(x_labels))],
        rotation=45, ha="right", fontsize=8)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Health Score (0-100)", fontsize=9)
    ax.set_title(f"{machine_id} — Health Score Trend & Forecast", fontsize=10, color=NAVY)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    fig.tight_layout(pad=0.8)
    return fig
