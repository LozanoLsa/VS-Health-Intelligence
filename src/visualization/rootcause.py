"""
Root Cause & Prescriptive visualizations.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from src.utils.logger import get_logger

log = get_logger(__name__)

NAVY   = "#0D1B2A"
WHITE  = "#FFFFFF"

TYPE_COLORS = {
    "mechanical": "#2980B9",
    "electrical": "#E74C3C",
    "software":   "#8E44AD",
    "hydraulic":  "#27AE60",
}

URGENCY_COLORS = {
    "Immediate":      "#E74C3C",
    "This week":      "#E67E22",
    "Within 2 weeks": "#F39C12",
    "Next PM cycle":  "#27AE60",
}


def plot_component_pareto(component_roi: pd.DataFrame,
                           top_n: int = 12) -> plt.Figure:
    """
    Pareto 80/20 bar chart: components ranked by total downtime cost.
    Left axis = cost ($), right axis = cumulative %.
    """
    df = component_roi.head(top_n).copy()
    labels = [f"{r['component']}\n({r['failure_type'][0].upper()})"
              for _, r in df.iterrows()]

    fig, ax1 = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(WHITE)
    ax1.set_facecolor("#F9F9F9")

    colors = [TYPE_COLORS.get(ft, "#999") for ft in df["failure_type"]]
    bars = ax1.bar(range(len(df)), df["total_cost"], color=colors, alpha=0.85,
                   edgecolor=NAVY, linewidth=0.6, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, df["total_cost"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"${val:,.0f}", ha="center", va="bottom",
                 fontsize=7, color=NAVY, fontweight="bold")

    # Pareto line (cumulative %)
    ax2 = ax1.twinx()
    ax2.plot(range(len(df)), df["cumul_pct"].values[:len(df)],
             color="#E84C1F", linewidth=2.2, marker="o",
             markersize=5, zorder=4, label="Cumulative %")
    ax2.axhline(80, color="#E84C1F", linestyle="--", linewidth=0.9, alpha=0.6)
    ax2.text(len(df) - 0.5, 81, "80%", color="#E84C1F", fontsize=8)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Cumulative % of Cost", fontsize=9)
    ax2.tick_params(labelsize=8)

    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("Total Downtime Cost ($)", fontsize=9)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax1.set_title("Component Cost Pareto — 80/20 Failure Intelligence",
                  fontsize=10, color=NAVY)

    legend_patches = [mpatches.Patch(facecolor=c, label=t.title())
                      for t, c in TYPE_COLORS.items()]
    ax1.legend(handles=legend_patches, loc="upper right",
               fontsize=7.5, framealpha=0.9, title="Failure Type")
    fig.tight_layout(pad=0.8)
    return fig


def plot_component_machine_heatmap(component_df: pd.DataFrame,
                                    vsm_filter: str = "All") -> plt.Figure:
    """
    Heatmap: machines (rows) × components (columns).
    Cell = total downtime hours for that combination.
    """
    df = component_df.copy()
    if vsm_filter != "All":
        df = df[df["vsm"] == vsm_filter]

    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data for selected filter",
                ha="center", va="center", fontsize=11, color="#888")
        ax.axis("off")
        return fig

    pivot = df.pivot_table(
        index="machine_id", columns="component",
        values="total_downtime_hrs", aggfunc="sum", fill_value=0
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2),
                                    max(5, len(pivot) * 0.45)))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    data = pivot.values
    vmax = max(data.max(), 1)
    im   = ax.imshow(data, cmap="YlOrRd", aspect="auto",
                     vmin=0, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Cell annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if val < vmax * 0.6 else "white",
                        fontweight="bold")

    plt.colorbar(im, ax=ax, label="Total Downtime Hours", shrink=0.8)
    ax.set_title(f"Component × Machine Downtime Heatmap"
                 + (f" — VSM {vsm_filter}" if vsm_filter != "All" else " — All VSMs"),
                 fontsize=10, color=NAVY)
    fig.tight_layout(pad=0.8)
    return fig


def plot_mttr_by_failure_type(component_df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar: avg MTTR per failure_type, one group per area.
    """
    df = component_df.copy()
    grp = (df.groupby(["area", "failure_type"])["mttr_component_hrs"]
             .mean().reset_index())
    areas       = sorted(grp["area"].unique())
    ftypes      = ["mechanical", "electrical", "software", "hydraulic"]
    x           = np.arange(len(areas))
    width       = 0.18
    offsets     = np.linspace(-(len(ftypes) - 1) / 2,
                               (len(ftypes) - 1) / 2, len(ftypes)) * width

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor("#F9F9F9")

    for offset, ft in zip(offsets, ftypes):
        vals = []
        for area in areas:
            row = grp[(grp["area"] == area) & (grp["failure_type"] == ft)]
            vals.append(float(row["mttr_component_hrs"].iloc[0]) if not row.empty else 0.0)
        bars = ax.bar(x + offset, vals, width=width * 0.92,
                      color=TYPE_COLORS.get(ft, "#999"),
                      label=ft.title(), alpha=0.85,
                      edgecolor=NAVY, linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=7, color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(areas, fontsize=9)
    ax.set_ylabel("Avg MTTR (hrs)", fontsize=9)
    ax.set_title("Avg MTTR by Failure Type & Area — Where is time being lost?",
                 fontsize=10, color=NAVY)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.set_ylim(0, grp["mttr_component_hrs"].max() * 1.35)
    fig.tight_layout(pad=0.8)
    return fig


def plot_prescriptive_urgency_chart(actions_df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart: machines ranked by cost_if_ignored,
    colored by urgency.
    """
    if actions_df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No prescriptive actions generated",
                ha="center", va="center", fontsize=11, color="#888")
        ax.axis("off")
        return fig

    df = (actions_df.groupby(["machine_id", "urgency"])
          .agg(cost_if_ignored=("cost_if_ignored", "sum"),
               cost_to_act=("cost_to_act", "sum"))
          .reset_index()
          .sort_values("cost_if_ignored", ascending=True))

    top = df.head(20)
    labels  = top["machine_id"].values
    ignored = top["cost_if_ignored"].values
    act     = top["cost_to_act"].values
    colors  = [URGENCY_COLORS.get(u, "#999") for u in top["urgency"].values]

    fig, ax = plt.subplots(figsize=(12, max(4, len(top) * 0.4)))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor("#F9F9F9")

    y = np.arange(len(top))
    ax.barh(y, ignored, color=colors, alpha=0.8,
            edgecolor=NAVY, linewidth=0.5, label="Cost if ignored (3 mo)")
    ax.barh(y, act, color="#B2BABB", alpha=0.85,
            edgecolor=NAVY, linewidth=0.5, label="Cost to act")

    for i, (ig, ac) in enumerate(zip(ignored, act)):
        ax.text(ig + 20, i, f"${ig:,.0f}", va="center",
                fontsize=7.5, color=NAVY, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Estimated Cost (USD)", fontsize=9)
    ax.set_title("Prescriptive Action — Cost to Act vs Cost if Ignored (3 months)",
                 fontsize=10, color=NAVY)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.6)

    urgency_patches = [mpatches.Patch(facecolor=c, label=u)
                       for u, c in URGENCY_COLORS.items()]
    ax.legend(handles=urgency_patches + [
        mpatches.Patch(facecolor="#B2BABB", label="Cost to act")],
        loc="lower right", fontsize=7.5, framealpha=0.9,
        title="Urgency")
    fig.tight_layout(pad=0.8)
    return fig
