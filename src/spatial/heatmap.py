"""
Heatmap engine — draws the plant layout with health-score color fills.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
import pandas as pd

from src.spatial.coordinates import load_zones, canvas_dimensions
from src.spatial.polygons import draw_rect, draw_area_band
from src.metrics.mtbf import HEALTH_COLORS
from src.utils.config import CONFIG
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

NAVY   = "#0D1B2A"
WHITE  = "#FFFFFF"
CREAM  = "#F7F3EE"
GRAY   = "#AAAAAA"
LIGHT  = "#E8E8E8"

AREA_COLORS = {
    "Raw Materials":  "#D6EAF8",
    "Machining":      "#EAF2FB",
    "Painting":       "#EBF5FB",
    "Assembly":       "#E8F8F5",
    "Finished Goods": "#D5F5E3",
}

VSM_LABEL_COLOR = {
    "Alpha": "#1A5276",
    "Beta":  "#1A5276",
    "Gamma": "#1A5276",
}

MACHINE_TYPE_SHORT = {
    "CNC":              "CNC",
    "Lathe":            "LT",
    "VMC":              "VMC",
    "Paint Booth":      "PB",
    "Curing Furnace":   "CF",
    "Assembly Station": "AS",
}


def _health_color(score: float) -> str:
    if score >= 70:
        return HEALTH_COLORS["Healthy"]
    elif score >= 40:
        return HEALTH_COLORS["Monitor"]
    return HEALTH_COLORS["Critical"]


def generate_heatmap(metrics_df: pd.DataFrame | None = None,
                     filter_vsm: str = "All",
                     filter_area: str = "All",
                     filter_status: str = "All",
                     output_path: str | None = None) -> plt.Figure:
    """
    Build a matplotlib Figure of the plant heatmap.
    Returns the Figure (for embedding in Streamlit) and saves PNG if output_path given.
    """
    if metrics_df is None:
        from src.etl.load import load_mtbf_metrics
        metrics_df = load_mtbf_metrics()

    zones = load_zones()
    canvas_w, canvas_h = canvas_dimensions()
    machines_z = zones["machines"]
    areas_z    = zones["areas"]
    aisles_z   = zones["aisles"]
    columns_z  = zones["columns"]
    warehouses = zones.get("warehouses", {})

    # Score lookup
    scores = dict(zip(metrics_df["machine_id"], metrics_df["health_score"]))
    statuses = dict(zip(metrics_df["machine_id"], metrics_df["health_status"]))
    mtbf_map = dict(zip(metrics_df["machine_id"], metrics_df["mtbf_hrs"]))
    mttr_map  = dict(zip(metrics_df["machine_id"], metrics_df["mttr_hrs"]))
    vsm_map   = dict(zip(metrics_df["machine_id"], metrics_df["vsm"]))
    area_map  = dict(zip(metrics_df["machine_id"], metrics_df["area"]))

    # Apply filters
    hidden = set()
    for mid in machines_z:
        vsm_val    = vsm_map.get(mid, "")
        area_val   = area_map.get(mid, "")
        status_val = statuses.get(mid, "")
        if filter_vsm    != "All" and vsm_val    != filter_vsm:    hidden.add(mid)
        if filter_area   != "All" and area_val   != filter_area:   hidden.add(mid)
        if filter_status != "All" and status_val != filter_status: hidden.add(mid)

    fig_w = 14
    fig_h = fig_w * (canvas_h / canvas_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Background: plant floor ──────────────────────────────────────────────
    ax.add_patch(Rectangle((0, 0), canvas_w, canvas_h,
                            facecolor=LIGHT, edgecolor=NAVY,
                            linewidth=2.0, zorder=0))

    # ── Warehouse bands ──────────────────────────────────────────────────────
    for wname, wz in warehouses.items():
        color = AREA_COLORS.get(wname.split()[0], "#DDD")
        ax.add_patch(Rectangle(
            (wz["x_min"], wz["y_min"]),
            wz["x_max"] - wz["x_min"],
            wz["y_max"] - wz["y_min"],
            facecolor=color, edgecolor=NAVY, linewidth=1.5,
            alpha=0.55, zorder=1))
        ax.text(canvas_w / 2, (wz["y_min"] + wz["y_max"]) / 2,
                wname.upper() + " WAREHOUSE",
                ha="center", va="center", fontsize=9,
                color=NAVY, fontweight="bold", alpha=0.7,
                fontfamily="monospace", zorder=2)

    # ── Area bands (horizontal stripes) ─────────────────────────────────────
    for aname, az in areas_z.items():
        color = AREA_COLORS.get(aname, "#EEE")
        ax.add_patch(Rectangle(
            (0, az["y_min"]), canvas_w, az["y_max"] - az["y_min"],
            facecolor=color, edgecolor=NAVY, linewidth=1.2,
            alpha=0.45, zorder=1))
        ax.text(0.25, (az["y_min"] + az["y_max"]) / 2, aname,
                ha="left", va="center", fontsize=8,
                color=NAVY, fontweight="bold", rotation=90,
                alpha=0.50, zorder=3)

    # ── Aisles ───────────────────────────────────────────────────────────────
    for aisle in aisles_z:
        ax.add_patch(Rectangle(
            (aisle["x"], 0), aisle["w"], canvas_h,
            facecolor="#CCCCCC", edgecolor=NAVY,
            linewidth=1.0, alpha=0.65, zorder=1))

    # ── VSM column headers ───────────────────────────────────────────────────
    for vsm_name, col in columns_z.items():
        cx = (col["x_min"] + col["x_max"]) / 2
        ax.text(cx, canvas_h - 0.25, f"VSM {vsm_name.upper()}",
                ha="center", va="top", fontsize=9.5,
                color=NAVY, fontweight="bold",
                fontfamily="monospace", zorder=5)

    # ── Machine boxes ────────────────────────────────────────────────────────
    for mid, mz in machines_z.items():
        score  = scores.get(mid, 50.0)
        status = statuses.get(mid, "Monitor")
        fcolor = _health_color(score)
        dimmed = mid in hidden

        alpha_box  = 0.18 if dimmed else 0.88
        alpha_text = 0.25 if dimmed else 1.0

        ax.add_patch(Rectangle(
            (mz["x"], mz["y"]), mz["w"], mz["h"],
            facecolor=fcolor if not dimmed else "#DDDDDD",
            edgecolor=NAVY,
            linewidth=1.2 if not dimmed else 0.5,
            alpha=alpha_box, zorder=3))

        cx = mz["x"] + mz["w"] / 2
        cy = mz["y"] + mz["h"] / 2

        # Machine ID
        ax.text(cx, cy + mz["h"] * 0.18, mid,
                ha="center", va="center", fontsize=8.5,
                color=NAVY, fontweight="bold",
                alpha=alpha_text, zorder=4)

        # Health score (large, center)
        ax.text(cx, cy - mz["h"] * 0.15, f"{score:.0f}",
                ha="center", va="center", fontsize=13,
                color=NAVY, fontweight="bold",
                alpha=alpha_text, zorder=4)

        # Status indicator dot (top-right corner)
        if not dimmed:
            dot_color = fcolor
            ax.plot(mz["x"] + mz["w"] - 0.18,
                    mz["y"] + mz["h"] - 0.22,
                    "o", ms=4, color=dot_color,
                    markeredgecolor=NAVY, markeredgewidth=0.4,
                    zorder=5)

    # ── Color legend ─────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=HEALTH_COLORS["Healthy"],
                       edgecolor=NAVY, label="Healthy  (70-100)"),
        mpatches.Patch(facecolor=HEALTH_COLORS["Monitor"],
                       edgecolor=NAVY, label="Monitor  (40-69)"),
        mpatches.Patch(facecolor=HEALTH_COLORS["Critical"],
                       edgecolor=NAVY, label="Critical  (0-39)"),
    ]
    ax.legend(handles=legend_patches,
              loc="lower right", fontsize=7.5,
              framealpha=0.92, edgecolor=NAVY,
              title="Health Score", title_fontsize=7.5,
              labelspacing=0.55)

    # ── KPI overlay text (top-left) ──────────────────────────────────────────
    healthy_n  = (metrics_df["health_status"] == "Healthy").sum()
    monitor_n  = (metrics_df["health_status"] == "Monitor").sum()
    critical_n = (metrics_df["health_status"] == "Critical").sum()
    summary = (f"Value Streams: {len(metrics_df)} machines  |  "
               f"Healthy {healthy_n}  Monitor {monitor_n}  Critical {critical_n}")
    ax.text(0.50, 1.025, summary,
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5,
            color=NAVY, alpha=0.80,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor=WHITE, alpha=0.85, edgecolor=GRAY))

    fig.tight_layout(pad=0.5)

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=CONFIG["heatmap"]["dpi"],
                    bbox_inches="tight", facecolor=WHITE)
        log.info(f"Heatmap saved -> {output_path}")

    return fig


def generate_heatmap_plotly(
        metrics_df: pd.DataFrame | None = None,
        filter_vsm: str = "All",
        filter_area: str = "All",
        filter_status: str = "All"):
    """
    Plotly version of the plant heatmap.
    Each machine box is hoverable — tooltip shows the health score breakdown
    (MTBF / Availability / Failures contributions).
    Uses fill='toself' + hoveron='fills' so the entire rectangle area triggers hover.
    """
    import plotly.graph_objects as go

    if metrics_df is None:
        from src.etl.load import load_mtbf_metrics
        metrics_df = load_mtbf_metrics()

    zones      = load_zones()
    canvas_w, canvas_h = canvas_dimensions()
    machines_z = zones["machines"]
    areas_z    = zones["areas"]
    aisles_z   = zones["aisles"]
    columns_z  = zones["columns"]
    warehouses = zones.get("warehouses", {})

    # ── Metric lookups ────────────────────────────────────────────────────────
    scores    = dict(zip(metrics_df["machine_id"], metrics_df["health_score"]))
    statuses  = dict(zip(metrics_df["machine_id"], metrics_df["health_status"]))
    mtbf_map  = dict(zip(metrics_df["machine_id"], metrics_df["mtbf_hrs"]))
    mttr_map  = dict(zip(metrics_df["machine_id"], metrics_df["mttr_hrs"]))
    avail_map = dict(zip(metrics_df["machine_id"], metrics_df["availability_pct"]))
    fails_map = dict(zip(metrics_df["machine_id"], metrics_df["failures_30d"]))
    vsm_map   = dict(zip(metrics_df["machine_id"], metrics_df["vsm"]))
    area_map  = dict(zip(metrics_df["machine_id"], metrics_df["area"]))
    cost_map  = dict(zip(metrics_df["machine_id"],
                         metrics_df["total_monthly_downtime_cost"]))

    # ── Score breakdown: how much each component contributes (pts) ────────────
    max_mtbf  = float(metrics_df["mtbf_hrs"].replace(999.9, float("nan")).max() or 1.0)
    max_fails = int(metrics_df["failures_30d"].max()) if metrics_df["failures_30d"].max() > 0 else 1

    def _contrib(row) -> tuple[float, float, float]:
        nm = min(float(row["mtbf_hrs"]), max_mtbf) / max_mtbf
        na = float(row["availability_pct"]) / 100.0
        nf = 1.0 - (float(row["failures_30d"]) / max_fails)
        return round(nm * 50, 1), round(na * 30, 1), round(nf * 20, 1)

    contrib = {row["machine_id"]: _contrib(row)
               for _, row in metrics_df.iterrows()}

    # ── Hidden machines (filters) ─────────────────────────────────────────────
    hidden: set[str] = set()
    for mid in machines_z:
        if filter_vsm    != "All" and vsm_map.get(mid, "")   != filter_vsm:   hidden.add(mid)
        if filter_area   != "All" and area_map.get(mid, "")  != filter_area:  hidden.add(mid)
        if filter_status != "All" and statuses.get(mid, "")  != filter_status: hidden.add(mid)

    # ── Build shapes (static background) ─────────────────────────────────────
    shapes = []

    # Plant floor
    shapes.append(dict(type="rect", x0=0, y0=0, x1=canvas_w, y1=canvas_h,
                       fillcolor=LIGHT, line=dict(color=NAVY, width=2),
                       layer="below"))

    # Warehouse bands
    for wname, wz in warehouses.items():
        color = AREA_COLORS.get(wname.split()[0], "#DDD")
        shapes.append(dict(type="rect",
                           x0=wz["x_min"], y0=wz["y_min"],
                           x1=wz["x_max"], y1=wz["y_max"],
                           fillcolor=color, opacity=0.55,
                           line=dict(color=NAVY, width=1.5), layer="below"))

    # Area bands
    for aname, az in areas_z.items():
        color = AREA_COLORS.get(aname, "#EEE")
        shapes.append(dict(type="rect",
                           x0=0, y0=az["y_min"],
                           x1=canvas_w, y1=az["y_max"],
                           fillcolor=color, opacity=0.45,
                           line=dict(color=NAVY, width=1.2), layer="below"))

    # Aisles
    for aisle in aisles_z:
        shapes.append(dict(type="rect",
                           x0=aisle["x"], y0=0,
                           x1=aisle["x"] + aisle["w"], y1=canvas_h,
                           fillcolor="#CCCCCC", opacity=0.65,
                           line=dict(color=NAVY, width=1.0), layer="below"))

    # Machine rectangles (colored by health)
    for mid, mz in machines_z.items():
        score  = scores.get(mid, 50.0)
        dimmed = mid in hidden
        fcolor = _health_color(score) if not dimmed else "#DDDDDD"
        alpha  = 0.18 if dimmed else 0.88
        shapes.append(dict(type="rect",
                           x0=mz["x"],           y0=mz["y"],
                           x1=mz["x"] + mz["w"], y1=mz["y"] + mz["h"],
                           fillcolor=fcolor, opacity=alpha,
                           line=dict(color=NAVY,
                                     width=1.2 if not dimmed else 0.5)))

    # ── Annotations (text labels — not hoverable) ─────────────────────────────
    annotations = []

    # Warehouse labels
    for wname, wz in warehouses.items():
        annotations.append(dict(
            x=canvas_w / 2,
            y=(wz["y_min"] + wz["y_max"]) / 2,
            text=f"<b>{wname.upper()} WAREHOUSE</b>",
            showarrow=False, opacity=0.7,
            font=dict(size=9, color=NAVY, family="monospace")))

    # Area labels (rotated)
    for aname, az in areas_z.items():
        annotations.append(dict(
            x=0.22, y=(az["y_min"] + az["y_max"]) / 2,
            text=f"<b>{aname}</b>",
            textangle=-90, showarrow=False, opacity=0.50,
            font=dict(size=8, color=NAVY)))

    # VSM column headers — pushed down from top edge for breathing room
    for vsm_name, col in columns_z.items():
        cx = (col["x_min"] + col["x_max"]) / 2
        annotations.append(dict(
            x=cx, y=canvas_h - 0.65,
            text=f"<b>VSM {vsm_name.upper()}</b>",
            showarrow=False,
            font=dict(size=15, color=NAVY, family="monospace")))

    # Machine ID + score text
    for mid, mz in machines_z.items():
        cx    = mz["x"] + mz["w"] / 2
        cy    = mz["y"] + mz["h"] / 2
        score = scores.get(mid, 50.0)
        dim_a = 0.25 if mid in hidden else 1.0

        annotations.append(dict(
            x=cx, y=cy + mz["h"] * 0.17,
            text=f"<b>{mid}</b>",
            showarrow=False, opacity=dim_a,
            font=dict(size=11, color=NAVY, family="monospace")))

        annotations.append(dict(
            x=cx, y=cy - mz["h"] * 0.17,
            text=f"<b>{score:.0f}</b>",
            showarrow=False, opacity=dim_a,
            font=dict(size=18, color=NAVY)))

    # ── Hover trace: one square marker per machine center ────────────────────
    # Canvas is 30x20 units; machines are ~2.7x1.9 units.
    # Marker size 58px ≈ 2 data units at typical render size → covers the box.
    fig = go.Figure()

    tip_xs, tip_ys, tip_texts = [], [], []

    for mid, mz in machines_z.items():
        if mid in hidden:
            continue

        score  = scores.get(mid, 50.0)
        mtbf   = mtbf_map.get(mid, 0.0)
        mttr   = mttr_map.get(mid, 0.0)
        avail  = avail_map.get(mid, 0.0)
        fails  = int(fails_map.get(mid, 0))
        vsm    = vsm_map.get(mid, "")
        area   = area_map.get(mid, "")
        cost   = cost_map.get(mid, 0.0)
        status = statuses.get(mid, "")
        c_mtbf, c_avail, c_fail = contrib.get(mid, (0.0, 0.0, 0.0))

        status_label = {"Healthy": "Healthy", "Monitor": "Monitor",
                        "Critical": "CRITICAL"}.get(status, status)

        # Score bar as simple pct text — no Unicode block chars (breaks on Windows)
        def _pct(pts, cap):
            return f"{pts:.1f} / {cap:.0f} pts  ({pts/cap*100:.0f}%)" if cap else "—"

        tip = (
            f"<b>{mid}</b>  |  VSM {vsm}  |  {area}<br>"
            f"Status: <b>{status_label}</b>   "
            f"Health Score: <b>{score:.1f} / 100</b><br>"
            f"<br>"
            f"<b>Score breakdown</b><br>"
            f"MTBF   x0.50 : {_pct(c_mtbf, 50)}<br>"
            f"Avail  x0.30 : {_pct(c_avail, 30)}<br>"
            f"Fails  x0.20 : {_pct(c_fail, 20)}<br>"
            f"<br>"
            f"MTBF:         <b>{mtbf:.0f} hrs</b><br>"
            f"MTTR:         <b>{mttr:.2f} hrs</b><br>"
            f"Availability: <b>{avail:.1f}%</b><br>"
            f"Failures 30d: <b>{fails}</b><br>"
            f"Monthly Cost: <b>${cost:,.0f}</b>"
        )

        tip_xs.append(mz["x"] + mz["w"] / 2)
        tip_ys.append(mz["y"] + mz["h"] / 2)
        tip_texts.append(tip)

    # Single trace — square markers, very low opacity (not zero!), large enough
    # to cover each machine box when hovering anywhere inside it.
    fig.add_trace(go.Scatter(
        x=tip_xs,
        y=tip_ys,
        mode="markers",
        marker=dict(
            symbol="square",
            size=58,                        # ~2 data units → covers machine box
            color="rgba(0,0,0,0.02)",       # nearly invisible but hover-active
            line=dict(width=0),
        ),
        hovertemplate="%{text}<extra></extra>",
        text=tip_texts,
        showlegend=False,
        name="",
    ))

    # ── Legend traces (visible squares) ──────────────────────────────────────
    for label, color in [
        ("Healthy  (≥ 70)", "#2ECC71"),
        ("Monitor (40–69)", "#F39C12"),
        ("Critical  (< 40)", "#E74C3C"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=14, color=color, symbol="square"),
            name=label, showlegend=True))

    # ── KPI summary ───────────────────────────────────────────────────────────
    healthy_n  = (metrics_df["health_status"] == "Healthy").sum()
    monitor_n  = (metrics_df["health_status"] == "Monitor").sum()
    critical_n = (metrics_df["health_status"] == "Critical").sum()
    summary    = (f"Value Streams: {len(metrics_df)} machines  |  "
                  f"🟢 Healthy {healthy_n}  🟡 Monitor {monitor_n}  🔴 Critical {critical_n}")

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(range=[0, canvas_w], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, canvas_h], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        margin=dict(l=10, r=10, t=40, b=55),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.04,
            xanchor="center", x=0.5,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=NAVY, borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=NAVY,
            font=dict(size=11, color=NAVY, family="monospace"),
        ),
        title=dict(
            text=summary,
            x=0.5, xanchor="center",
            font=dict(size=10, color=NAVY),
        ),
    )

    return fig


def build_and_save() -> str:
    """Convenience wrapper called by ETL runner and Streamlit refresh."""
    from src.etl.load import load_mtbf_metrics
    metrics = load_mtbf_metrics()
    out = CONFIG["heatmap"]["output_path"]
    generate_heatmap(metrics, output_path=out)
    return out
