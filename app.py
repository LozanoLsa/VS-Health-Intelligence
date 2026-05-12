"""
Value Stream Health Intelligence Dashboard
Streamlit entry point  —  run: streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import io
import pandas as pd
import streamlit as st

from src.spatial.heatmap import generate_heatmap, generate_heatmap_plotly
from src.etl.load import load_mtbf_metrics, load_monthly_metrics, load_component_metrics
from src.visualization.plots import vsm_summary_table, top_critical, top_healthy
from src.visualization.trends import (
    plot_vsm_monthly_trend,
    plot_monthly_failures_bar,
    plot_health_calendar,
    plot_machine_forecast,
)
from src.visualization.rootcause import (
    plot_component_pareto,
    plot_component_machine_heatmap,
    plot_mttr_by_failure_type,
    plot_prescriptive_urgency_chart,
)
from src.ml.forecasting import generate_machine_forecasts, vsm_forecast_summary, forecasts_to_df
from src.ml.explainer import generate_machine_insight, generate_fleet_risk_table
from src.ml.prescriptive import (
    generate_prescriptive_actions,
    generate_executive_summary,
    compute_component_roi_table,
    actions_to_df,
)
from src.utils.logger import get_logger

log = get_logger("app")


def _r(df: pd.DataFrame) -> pd.DataFrame:
    """Round all numeric columns to 2 decimal places before display."""
    num_cols = df.select_dtypes(include="number").columns
    return df.assign(**{c: df[c].round(2) for c in num_cols})


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Value Stream Health Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "metrics" not in st.session_state:
    st.session_state.metrics = load_mtbf_metrics()
if "monthly" not in st.session_state:
    try:
        st.session_state.monthly = load_monthly_metrics()
    except Exception:
        st.session_state.monthly = None
if "components" not in st.session_state:
    try:
        st.session_state.components = load_component_metrics()
    except Exception:
        st.session_state.components = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h3 style='color:#0D1B2A; margin-bottom:4px;'>VS Filters</h3>",
        unsafe_allow_html=True,
    )

    filter_vsm = st.selectbox(
        "VSM Line", ["All", "Alpha", "Beta", "Gamma"], index=0)
    filter_area = st.selectbox(
        "Area", ["All", "Machining", "Painting", "Assembly"], index=0)
    filter_status = st.selectbox(
        "Health Status", ["All", "Healthy", "Monitor", "Critical"], index=0)

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#0D1B2A; margin-bottom:4px;'>Forecast Settings</h3>",
        unsafe_allow_html=True,
    )

    forecast_horizon = st.radio(
        "Forecast Horizon", [1, 2, 3],
        format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
        horizontal=True,
    )
    alert_threshold = st.slider(
        "Alert Threshold (Health Score)", min_value=30, max_value=70,
        value=55, step=5,
        help="Machines forecast below this score will be flagged as MEDIUM risk",
    )

    st.markdown("---")
    st.caption("LozanoLsa · Turning Operations into Predictive Systems")


# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding:10px 0 4px;">
      <h2 style="margin:0; color:#0D1B2A; letter-spacing:1px; font-size:1.55rem;">
        Value Stream Health Intelligence
        &nbsp;&mdash;&nbsp;
        MTBF &middot; MTTR &middot; OEE &middot; Predictive Analytics
      </h2>
      <p style="margin:5px 0 0; color:#555; font-size:0.88rem;">
        Spatial heatmapping &nbsp;&middot;&nbsp; Time-series forecasting
        &nbsp;&middot;&nbsp; Operational Excellence reliability metrics
      </p>
    </div>
    <hr style="border:none; border-top:2px solid #E84C1F; margin:8px 0 14px;">
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
metrics: pd.DataFrame = st.session_state.metrics
monthly:    pd.DataFrame | None = st.session_state.monthly
components: pd.DataFrame | None = st.session_state.components

filtered = metrics.copy()
if filter_vsm    != "All": filtered = filtered[filtered["vsm"]           == filter_vsm]
if filter_area   != "All": filtered = filtered[filtered["area"]          == filter_area]
if filter_status != "All": filtered = filtered[filtered["health_status"] == filter_status]

# ── Pre-compute VSM forecast summary (used in Tab 1 + Tab 3) ─────────────────
_vsm_fc_summary = None
if monthly is not None:
    try:
        _fc_all = generate_machine_forecasts(monthly, horizon=forecast_horizon)
        _vsm_fc_summary = vsm_forecast_summary(_fc_all)
    except Exception:
        _vsm_fc_summary = None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "VS Overview",
    "Monthly Trends",
    "Predictive Simulation",
    "Root Cause & Prescriptive",
])

# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 1 — FLEET OVERVIEW
# ╚══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Period selector ───────────────────────────────────────────────────────
    _avail_months = sorted(monthly["year_month"].unique().tolist()) if monthly is not None else []
    _period_opts  = ["All time (overall)"] + _avail_months
    sel_col, _ = st.columns([0.32, 0.68])
    with sel_col:
        selected_period = st.selectbox(
            "🗓 Heatmap period",
            _period_opts,
            index=0,
            help=(
                "'All time' shows KPIs calculated across the full history. "
                "Select a specific month to see the plant health state at that point in time."
            ),
        )

    # Build heatmap_metrics for the selected period
    if selected_period == "All time (overall)" or monthly is None:
        heatmap_metrics   = metrics
        period_label      = "Overall"
        show_fc_cards     = True
    else:
        _mdf = monthly[monthly["year_month"] == selected_period].copy()
        _mdf = _mdf.rename(columns={
            "failures_count":     "failures_30d",
            "monthly_downtime_cost": "total_monthly_downtime_cost",
        })
        heatmap_metrics = _mdf
        period_label    = selected_period
        show_fc_cards   = False

    # ── Health formula popover ────────────────────────────────────────────────
    col_title, col_pop = st.columns([0.88, 0.12])
    with col_title:
        _period_sfx = f" — {period_label}" if period_label != "Overall" else ""
        st.markdown(f"#### Plant Heatmap — Health Scores{_period_sfx}")
    with col_pop:
        with st.popover("? Formula"):
            st.markdown(
                """
                **Health Score formula**

                ```
                Health = (MTBF_norm × 0.50)
                       + (Availability × 0.30)
                       + (1 − Failures_norm × 0.20)
                × 100
                ```
                | Zone | Score | Color |
                |------|-------|-------|
                | Healthy  | ≥ 70 | 🟢 |
                | Monitor  | 40–69 | 🟡 |
                | Critical | < 40 | 🔴 |

                *Normalization caps:*  MTBF = 744 hrs (31-day max),
                Availability = 100%, Failures = VS monthly max.
                """
            )

    # ── Heatmap (Plotly — interactive hover) ─────────────────────────────────
    with st.spinner("Rendering heatmap..."):
        fig_heat = generate_heatmap_plotly(
            metrics_df=heatmap_metrics,
            filter_vsm=filter_vsm,
            filter_area=filter_area,
            filter_status=filter_status,
        )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── How to read the Health Score ─────────────────────────────────────────
    with st.expander("🩺 How to read a machine's health score"):
        st.markdown(
            """
            <div style="font-size:0.93rem; color:#0D1B2A; line-height:1.7;">

            Think of the Health Score as a <b>doctor's diagnosis</b> —
            not a single test, but three questions asked at once,
            each weighted by how much it reveals about the machine's real condition.

            </div>
            """,
            unsafe_allow_html=True,
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            st.markdown(
                """
                <div style="background:#EAF2FB; border-left:5px solid #1A5276;
                            border-radius:8px; padding:14px 16px; height:100%;">
                  <div style="font-size:1.05rem; font-weight:700; color:#1A5276;">
                    ① How long does it run<br>before breaking?
                  </div>
                  <div style="font-size:0.78rem; font-weight:700; color:#555;
                              margin:6px 0 10px; letter-spacing:0.5px;">
                    MTBF &nbsp;·&nbsp; weight 50%
                  </div>
                  <div style="font-size:0.88rem; color:#333;">
                    The strongest reliability signal.
                    A machine that runs 400 hrs between failures
                    is fundamentally different from one that runs 40 hrs —
                    no matter how fast it gets repaired.
                    <br><br>
                    <b>Gets the most weight (50%)</b> because it reflects
                    what's happening <i>inside</i> the machine.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with q2:
            st.markdown(
                """
                <div style="background:#E8F8F5; border-left:5px solid #117A65;
                            border-radius:8px; padding:14px 16px; height:100%;">
                  <div style="font-size:1.05rem; font-weight:700; color:#117A65;">
                    ② When it breaks, how much<br>production is lost?
                  </div>
                  <div style="font-size:0.78rem; font-weight:700; color:#555;
                              margin:6px 0 10px; letter-spacing:0.5px;">
                    AVAILABILITY &nbsp;·&nbsp; weight 30%
                  </div>
                  <div style="font-size:0.88rem; color:#333;">
                    What the Plant Manager feels directly on the floor.
                    A machine at 97% availability lost only 2.4 hrs
                    out of every 100 hrs of planned production.
                    <br><br>
                    <b>30% weight</b> because downtime is the
                    cost the business actually pays.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with q3:
            st.markdown(
                """
                <div style="background:#FEF9E7; border-left:5px solid #D4AC0D;
                            border-radius:8px; padding:14px 16px; height:100%;">
                  <div style="font-size:1.05rem; font-weight:700; color:#B7950B;">
                    ③ Is it getting worse<br>recently?
                  </div>
                  <div style="font-size:0.78rem; font-weight:700; color:#555;
                              margin:6px 0 10px; letter-spacing:0.5px;">
                    FAILURES (30 days) &nbsp;·&nbsp; weight 20%
                  </div>
                  <div style="font-size:0.88rem; color:#333;">
                    The early-warning signal. A machine can have
                    a good historical MTBF but suddenly fail
                    3 times this month — that shift doesn't show
                    up in the other two numbers.
                    <br><br>
                    <b>20% weight</b> adds recency context
                    that MTBF alone misses.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:#F4F6F7; border-radius:8px; padding:14px 18px;
                        border-left:5px solid #E84C1F; font-size:0.9rem; color:#0D1B2A;">
              <b>🔑 The key insight:</b> &nbsp;No machine is judged against an absolute standard.
              Each score is normalized against the <b>best performer in its own Value Stream</b>.
              A CNC in Gamma competes only against other Gamma machines — not against Alpha's best.
              This makes the score <b>fair, contextual, and actionable</b> regardless of machine type or age.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📋 Real examples from this dataset")
        ex1, ex2 = st.columns(2)

        with ex1:
            st.markdown(
                """
                <div style="background:#EAFAF1; border:1px solid #2ECC71;
                            border-radius:8px; padding:14px 16px;">
                  <div style="font-size:1.0rem; font-weight:700; color:#1E8449;">
                    CF-A1 &nbsp;·&nbsp;
                    <span style="background:#2ECC71; color:white; border-radius:4px;
                                 padding:2px 8px; font-size:0.8rem;">Healthy · 74.8</span>
                  </div>
                  <div style="font-size:0.8rem; color:#555; margin:4px 0 10px;">
                    VSM Alpha · Painting
                  </div>
                  <table style="font-size:0.85rem; width:100%; border-collapse:collapse;">
                    <tr style="color:#555;">
                      <td style="padding:3px 0;"><b>① MTBF</b></td>
                      <td>247 hrs &nbsp;÷&nbsp; 400 hrs best</td>
                      <td style="text-align:right;"><b>30.5 / 50</b></td>
                    </tr>
                    <tr style="color:#555;">
                      <td style="padding:3px 0;"><b>② Avail.</b></td>
                      <td>97.6% &nbsp;→&nbsp; near perfect</td>
                      <td style="text-align:right;"><b>29.3 / 30</b></td>
                    </tr>
                    <tr style="color:#555;">
                      <td style="padding:3px 0;"><b>③ Fails</b></td>
                      <td>4 this month &nbsp;÷&nbsp; 20 max in VS</td>
                      <td style="text-align:right;"><b>15.0 / 20</b></td>
                    </tr>
                    <tr style="border-top:1px solid #ccc;">
                      <td colspan="2" style="padding-top:6px; font-weight:700;">Score</td>
                      <td style="text-align:right; font-weight:700; color:#1E8449;
                                 font-size:1.1rem; padding-top:6px;">74.8</td>
                    </tr>
                  </table>
                  <div style="font-size:0.8rem; color:#555; margin-top:8px; font-style:italic;">
                    Reliable, minimal downtime, low recent failures.
                    The 15/20 on failures hints at something worth watching,
                    but overall this machine is well-behaved.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ex2:
            st.markdown(
                """
                <div style="background:#FDECEA; border:1px solid #E74C3C;
                            border-radius:8px; padding:14px 16px;">
                  <div style="font-size:1.0rem; font-weight:700; color:#922B21;">
                    LT-B1 &nbsp;·&nbsp;
                    <span style="background:#E74C3C; color:white; border-radius:4px;
                                 padding:2px 8px; font-size:0.8rem;">Critical · 38.8</span>
                  </div>
                  <div style="font-size:0.8rem; color:#555; margin:4px 0 10px;">
                    VSM Beta · Machining
                  </div>
                  <table style="font-size:0.85rem; width:100%; border-collapse:collapse;">
                    <tr style="color:#555;">
                      <td style="padding:3px 0;"><b>① MTBF</b></td>
                      <td>60 hrs &nbsp;÷&nbsp; 400 hrs best</td>
                      <td style="text-align:right;"><b style="color:#E74C3C;">7.5 / 50</b></td>
                    </tr>
                    <tr style="color:#555;">
                      <td style="padding:3px 0;"><b>② Avail.</b></td>
                      <td>87.9% &nbsp;→&nbsp; losing production</td>
                      <td style="text-align:right;"><b style="color:#F39C12;">26.4 / 30</b></td>
                    </tr>
                    <tr style="color:#555;">
                      <td style="padding:3px 0;"><b>③ Fails</b></td>
                      <td>12 this month &nbsp;÷&nbsp; 20 max in VS</td>
                      <td style="text-align:right;"><b style="color:#E74C3C;">5.0 / 20</b></td>
                    </tr>
                    <tr style="border-top:1px solid #ccc;">
                      <td colspan="2" style="padding-top:6px; font-weight:700;">Score</td>
                      <td style="text-align:right; font-weight:700; color:#E74C3C;
                                 font-size:1.1rem; padding-top:6px;">38.8</td>
                    </tr>
                  </table>
                  <div style="font-size:0.8rem; color:#555; margin-top:8px; font-style:italic;">
                    Failing every 60 hrs, losing 12% of production time,
                    and getting worse this month. Each of the three signals
                    is red — the score is the sum of three bad answers.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── VSM Cost Intelligence Cards ───────────────────────────────────────────
    st.markdown("#### Value Stream Cost Intelligence")

    vsm_colors   = {"Alpha": "#1A5276", "Beta": "#117A65", "Gamma": "#7B241C"}
    vsm_bg       = {"Alpha": "#EAF2FB", "Beta": "#E8F8F5", "Gamma": "#FDEDEC"}
    vsm_machines = {"Alpha": 12, "Beta": 12, "Gamma": 11}

    _health_now_label = "avg health (now)" if period_label == "Overall" else f"avg health ({period_label})"

    c1, c2, c3 = st.columns(3)
    for col, vsm in zip([c1, c2, c3], ["Alpha", "Beta", "Gamma"]):
        vsm_data  = heatmap_metrics[heatmap_metrics["vsm"] == vsm]
        n_total   = len(vsm_data)
        n_crit    = int((vsm_data["health_status"] == "Critical").sum())
        n_mon     = int((vsm_data["health_status"] == "Monitor").sum())
        n_ok      = int((vsm_data["health_status"] == "Healthy").sum())
        avg_score = float(vsm_data["health_score"].mean())
        avg_avail = float(vsm_data["availability_pct"].mean())
        avg_mtbf  = float(vsm_data["mtbf_hrs"].mean())
        avg_mttr  = float(vsm_data["mttr_hrs"].mean())
        cost_mo   = float(vsm_data["total_monthly_downtime_cost"].sum())

        border_col = "#E74C3C" if n_crit > 0 else ("#F39C12" if n_mon > 0 else "#2ECC71")
        score_col  = "#2ECC71" if avg_score >= 70 else ("#F39C12" if avg_score >= 40 else "#E74C3C")

        # Forecast variables (all plain Python — no nested f-strings)
        fc_score_val  = 0.0
        fc_score_col  = "#888"
        fc_delta_str  = ""
        fc_delta_col  = "#888"
        fc_slope_str  = ""
        fc_slope_col  = "#888"
        show_fc       = False

        if _vsm_fc_summary is not None:
            fc_row = _vsm_fc_summary[_vsm_fc_summary["vsm"] == vsm]
            if not fc_row.empty:
                show_fc      = True
                fc_score_val = float(fc_row["avg_health_score"].iloc[0])
                fc_slope_val = float(fc_row["avg_slope"].iloc[0])
                delta        = fc_score_val - avg_score
                fc_score_col = "#2ECC71" if fc_score_val >= 70 else ("#F39C12" if fc_score_val >= 40 else "#E74C3C")
                fc_delta_col = "#2ECC71" if delta >= 0 else "#E74C3C"
                fc_delta_str = ("+" if delta >= 0 else "") + f"{delta:.1f}"
                fc_slope_col = "#2ECC71" if fc_slope_val >= 0 else "#E74C3C"
                fc_slope_str = ("+" if fc_slope_val >= 0 else "") + f"{fc_slope_val:.2f} pts/mo"

        col.markdown(
            f"""
            <div style="background:{vsm_bg[vsm]}; border-left:5px solid {border_col};
                        border-radius:8px; padding:14px 16px 10px;">

              <div style="font-size:1.0rem; font-weight:700;
                          color:{vsm_colors[vsm]}; margin-bottom:8px;">
                VSM {vsm}
                <span style="font-size:0.8rem; color:#555; font-weight:400;">
                  &nbsp;({n_total} machines)
                </span>
              </div>

              <div style="font-size:1.6rem; font-weight:700; color:{score_col};">
                {avg_score:.1f}
                <span style="font-size:0.8rem; color:#555; font-weight:400;">
                  &nbsp;{_health_now_label}
                </span>
              </div>

              <div style="font-size:0.82rem; color:#333; margin-top:8px;
                          display:grid; grid-template-columns:1fr 1fr; gap:4px 10px;">
                <div><b>Healthy:</b> {n_ok}</div>
                <div><b>Monitor:</b> {n_mon}</div>
                <div><b>Critical:</b>
                  <span style="color:#E74C3C; font-weight:700;">{n_crit}</span>
                </div>
                <div><b>Avail.:</b> {avg_avail:.1f}%</div>
                <div><b>MTBF:</b> {avg_mtbf:.0f} hrs</div>
                <div><b>MTTR:</b> {avg_mttr:.2f} hrs</div>
              </div>

              <div style="margin-top:8px; font-size:0.82rem;">
                <b>Monthly Cost:</b>
                <span style="font-weight:700; color:#0D1B2A;">${cost_mo:,.0f}</span>
              </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # Forecast block rendered separately to avoid nested f-string bug
        if show_fc and show_fc_cards:
            col.markdown(
                f"""
                <div style="background:{vsm_bg[vsm]}; border-left:5px solid {fc_score_col};
                            border-radius:8px; padding:10px 16px; margin-top:4px;">
                  <div style="font-size:0.8rem; color:#555; font-weight:600; margin-bottom:4px;">
                    Forecast +{forecast_horizon}mo
                  </div>
                  <div style="font-size:1.6rem; font-weight:700; color:{fc_score_col};">
                    {fc_score_val:.1f}
                    <span style="font-size:0.8rem; color:{fc_delta_col}; font-weight:600;">
                      &nbsp;({fc_delta_str})
                    </span>
                  </div>
                  <div style="font-size:0.82rem; margin-top:4px;">
                    <b>Trend:</b>
                    <span style="color:{fc_slope_col}; font-weight:700;">{fc_slope_str}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Critical & Healthy tables ─────────────────────────────────────────────
    col_crit, col_ok = st.columns(2)

    with col_crit:
        st.markdown("##### Top Critical Machines")
        crit_df = top_critical(heatmap_metrics, n=5)
        if crit_df.empty:
            st.success("No critical machines — all above threshold!")
        else:
            st.dataframe(
                _r(crit_df).style.set_properties(**{
                    "background-color": "#FDECEA", "color": "#7B241C"}),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Score":       st.column_config.NumberColumn(format="%.2f"),
                    "MTBF (hrs)":  st.column_config.NumberColumn(format="%.2f"),
                    "MTTR (hrs)":  st.column_config.NumberColumn(format="%.2f"),
                    "Fails (30d)": st.column_config.NumberColumn(format="%d"),
                },
            )

    with col_ok:
        st.markdown("##### Top Healthiest Machines")
        healthy_df = top_healthy(heatmap_metrics, n=5)
        st.dataframe(
            _r(healthy_df).style.set_properties(**{
                "background-color": "#EAFAF1", "color": "#1E8449"}),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score":            st.column_config.NumberColumn(format="%.2f"),
                "MTBF (hrs)":       st.column_config.NumberColumn(format="%.2f"),
                "MTTR (hrs)":       st.column_config.NumberColumn(format="%.2f"),
                "Availability (%)": st.column_config.NumberColumn(format="%.2f"),
                "Fails (30d)":      st.column_config.NumberColumn(format="%d"),
            },
        )

    st.markdown("---")

    # ── Download ──────────────────────────────────────────────────────────────
    csv_buf = io.StringIO()
    filtered.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download Filtered Metrics CSV",
        data=csv_buf.getvalue(),
        file_name="mtbf_metrics_filtered.csv",
        mime="text/csv",
    )


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 2 — MONTHLY TRENDS
# ╚══════════════════════════════════════════════════════════════════════════════
with tab2:

    if monthly is None:
        st.warning(
            "Monthly metrics not found. Click **Refresh Data** in the sidebar "
            "to generate `monthly_metrics.csv` from the ETL pipeline."
        )
    else:
        monthly_f = monthly.copy()
        if filter_vsm  != "All": monthly_f = monthly_f[monthly_f["vsm"]  == filter_vsm]
        if filter_area != "All": monthly_f = monthly_f[monthly_f["area"] == filter_area]

        # ── Date range + metric selector ──────────────────────────────────────
        all_months = sorted(monthly_f["year_month"].unique())

        col_from, col_to, col_metric = st.columns([0.28, 0.28, 0.44])
        with col_from:
            month_from = st.selectbox(
                "📅 From", all_months,
                index=0,
                key="month_from_sel",
            )
        with col_to:
            # Only allow end months >= start month
            valid_to = [m for m in all_months if m >= month_from]
            month_to = st.selectbox(
                "📅 To", valid_to,
                index=len(valid_to) - 1,
                key="month_to_sel",
            )
        with col_metric:
            metric_options = {
                "Health Score":         "health_score",
                "MTBF (hrs)":           "mtbf_hrs",
                "MTTR (hrs)":           "mttr_hrs",
                "Availability (%)":     "availability_pct",
                "Monthly Downtime ($)": "monthly_downtime_cost",
            }
            selected_label = st.selectbox(
                "📊 Metric to plot", list(metric_options.keys()), index=0)
            metric_col = metric_options[selected_label]

        # Filter all data to selected range
        in_range = (monthly_f["year_month"] >= month_from) & \
                   (monthly_f["year_month"] <= month_to)
        monthly_range = monthly_f[in_range]
        n_months_sel  = monthly_range["year_month"].nunique()

        st.caption(
            f"Range: **{month_from}** → **{month_to}** "
            f"— {n_months_sel} month{'s' if n_months_sel != 1 else ''} selected"
        )

        # ── KPI summary for selected range ────────────────────────────────────
        _total_failures = int(monthly_range["failures_count"].sum())
        _total_cost     = monthly_range["monthly_downtime_cost"].sum()
        _avg_fails_mo   = _total_failures / n_months_sel if n_months_sel > 0 else 0
        _avg_cost_mo    = _total_cost     / n_months_sel if n_months_sel > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Machines",
            monthly_range["machine_id"].nunique(),
            help="Number of unique machines with at least one record in the selected date range.",
        )
        k2.metric(
            "Avg Score",
            f"{monthly_range['health_score'].mean():.1f}",
            help=(
                "Average health score across all machine-months in the selected range. "
                "Each machine contributes one score per month (0–100). "
                "This is the Value Stream-wide average over the entire period."
            ),
        )
        k3.metric(
            "Total Failures",
            f"{_total_failures:,}",
            help=(
                f"Sum of ALL failure events recorded across all {monthly_range['machine_id'].nunique()} machines "
                f"over the {n_months_sel} selected months. "
                f"This is an accumulated count — not a snapshot. "
                f"Monthly average: ~{_avg_fails_mo:.0f} failures / month across all Value Streams."
            ),
        )
        k4.metric(
            "Total Cost",
            f"${_total_cost:,.0f}",
            help=(
                f"Accumulated downtime cost over the {n_months_sel} selected months. "
                f"Calculated as: downtime hours × cost per hour, per machine, per month — then summed. "
                f"Monthly average: ~${_avg_cost_mo:,.0f} / month. "
                f"Source: equipment_master.csv (downtime_cost_per_hr) × failures.csv (downtime_hrs)."
            ),
        )
        st.caption(
            f"↑ All four KPIs are **accumulated totals** over the {n_months_sel}-month window "
            f"({month_from} → {month_to}), not point-in-time snapshots. "
            f"Monthly averages: **~{_avg_fails_mo:.0f} failures/mo** · "
            f"**~${_avg_cost_mo:,.0f}/mo** downtime cost."
        )

        st.markdown("---")

        # ── Line trend chart ──────────────────────────────────────────────────
        st.markdown("#### VSM Monthly Trend")
        fig_trend = plot_vsm_monthly_trend(
            monthly_range, metric_col=metric_col, vsm_filter="All")
        st.pyplot(fig_trend, use_container_width=True)

        # ── Stacked failures bar (full range) ────────────────────────────────
        st.markdown("#### Monthly Failure Count by Area")
        monthly_bar = monthly.copy()
        if filter_vsm != "All":
            monthly_bar = monthly_bar[monthly_bar["vsm"] == filter_vsm]
        monthly_bar = monthly_bar[
            (monthly_bar["year_month"] >= month_from) &
            (monthly_bar["year_month"] <= month_to)
        ]
        fig_bar = plot_monthly_failures_bar(monthly_bar)
        st.pyplot(fig_bar, use_container_width=True)

        # ── Health score calendar (filtered to range) ─────────────────────────
        st.markdown("#### Health Score Calendar (machine × month)")
        st.caption(
            f"Showing {month_from} → {month_to}. "
            "Red = Critical  ·  Amber = Monitor  ·  Green = Healthy."
        )
        pivot = plot_health_calendar(monthly_range)

        def _color_score(val):
            if pd.isna(val): return ""
            if val >= 70:  return "background-color:#d4efdf; color:#1E8449;"
            if val >= 40:  return "background-color:#fef9e7; color:#9A7D0A;"
            return "background-color:#FDECEA; color:#7B241C;"

        def _highlight_month(col):
            if col.name == month_to:
                return ["font-weight:700; border-left:3px solid #0D1B2A;"] * len(col)
            return [""] * len(col)

        pivot_r = _r(pivot)
        month_cfg = {c: st.column_config.NumberColumn(format="%.2f")
                     for c in pivot_r.columns}
        st.dataframe(
            pivot_r.style
                .map(_color_score)
                .apply(_highlight_month, axis=0),
            use_container_width=True,
            column_config=month_cfg,
        )

        st.markdown("---")

        # ── Snapshot: last month of range ─────────────────────────────────────
        st.markdown(f"#### Machine Snapshot — {month_to}")
        st.caption(f"KPI detail for every machine in {month_to}, sorted worst → best score.")

        snap = monthly_f[monthly_f["year_month"] == month_to].copy()

        snap_disp = _r(snap[[
            "machine_id", "vsm", "area", "machine_type",
            "health_score", "health_status",
            "mtbf_hrs", "mttr_hrs", "availability_pct",
            "failures_count", "downtime_hrs", "monthly_downtime_cost",
        ]].sort_values("health_score").rename(columns={
            "machine_id":            "Machine",
            "vsm":                   "VSM",
            "area":                  "Area",
            "machine_type":          "Type",
            "health_score":          "Score",
            "health_status":         "Status",
            "mtbf_hrs":              "MTBF (hrs)",
            "mttr_hrs":              "MTTR (hrs)",
            "availability_pct":      "Avail. (%)",
            "failures_count":        "Failures",
            "downtime_hrs":          "Downtime (hrs)",
            "monthly_downtime_cost": "Cost ($)",
        }))

        def _color_status(val):
            if val == "Critical": return "background-color:#FDECEA; color:#7B241C; font-weight:700;"
            if val == "Monitor":  return "background-color:#fef9e7; color:#9A7D0A; font-weight:600;"
            return "background-color:#EAFAF1; color:#1E8449;"

        st.dataframe(
            snap_disp.style
                .map(_color_score,  subset=["Score"])
                .map(_color_status, subset=["Status"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score":          st.column_config.NumberColumn(format="%.2f"),
                "MTBF (hrs)":     st.column_config.NumberColumn(format="%.2f"),
                "MTTR (hrs)":     st.column_config.NumberColumn(format="%.2f"),
                "Avail. (%)":     st.column_config.NumberColumn(format="%.2f"),
                "Failures":       st.column_config.NumberColumn(format="%d"),
                "Downtime (hrs)": st.column_config.NumberColumn(format="%.2f"),
                "Cost ($)":       st.column_config.NumberColumn(format="$%.2f"),
            },
        )


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 3 — PREDICTIVE SIMULATION
# ╚══════════════════════════════════════════════════════════════════════════════
with tab3:

    if monthly is None:
        st.warning(
            "Monthly metrics not found. Click **Refresh Data** in the sidebar "
            "to generate `monthly_metrics.csv` from the ETL pipeline."
        )
    else:
        # ── Executive pitch banner ────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="
              background: linear-gradient(135deg, #0D1B2A 0%, #1A3552 100%);
              border-radius: 10px;
              padding: 18px 24px;
              margin-bottom: 18px;
              color: white;
            ">
              <div style="font-size:1.1rem; font-weight:700; letter-spacing:0.5px;">
                Predictive Simulation &nbsp;—&nbsp; Horizon: +{forecast_horizon} month{'s' if forecast_horizon > 1 else ''}
              </div>
              <div style="font-size:0.85rem; color:#A9CCE3; margin-top:6px;">
                Linear trend models with 80% prediction intervals &nbsp;&middot;&nbsp;
                Value Stream: {len(metrics)} machines &nbsp;&middot;&nbsp;
                Alert threshold: {alert_threshold} pts
              </div>
              <div style="font-size:0.8rem; color:#85C1E9; margin-top:6px; font-style:italic;">
                Each machine has an independently fitted trend on its health score time series.
                Residual standard error drives the confidence band — no black box, fully auditable.
                Value Stream view: Alpha &middot; Beta &middot; Gamma.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Use pre-computed forecasts ────────────────────────────────────────
        with st.spinner("Generating ML forecasts..."):
            forecasts = _fc_all if _vsm_fc_summary is not None else \
                        generate_machine_forecasts(monthly, horizon=forecast_horizon)
            vsm_fc_df = _vsm_fc_summary if _vsm_fc_summary is not None else \
                        vsm_forecast_summary(forecasts)

        # ── VSM Forecast Cards ────────────────────────────────────────────────
        st.markdown("#### Value Stream Forecast Summary")
        fc1, fc2, fc3 = st.columns(3)
        for col, vsm in zip([fc1, fc2, fc3], ["Alpha", "Beta", "Gamma"]):
            row = vsm_fc_df[vsm_fc_df["vsm"] == vsm]
            if row.empty:
                continue
            r = row.iloc[0]
            slope_sign = "+" if r["avg_slope"] >= 0 else ""
            slope_color = "#2ECC71" if r["avg_slope"] >= 0 else "#E74C3C"
            score_color = "#2ECC71" if r["avg_health_score"] >= 70 \
                else ("#F39C12" if r["avg_health_score"] >= 40 else "#E74C3C")
            bg_color = vsm_bg[vsm]
            col.markdown(
                f"""
                <div style="
                  background:{bg_color};
                  border-left: 5px solid {score_color};
                  border-radius: 8px;
                  padding: 14px 16px 10px;
                ">
                  <div style="font-size:1.0rem; font-weight:700;
                              color:{vsm_colors[vsm]}; margin-bottom:8px;">
                    VSM {vsm} &nbsp;
                    <span style="font-size:0.8rem; color:#555; font-weight:400;">
                      +{forecast_horizon}mo forecast
                    </span>
                  </div>
                  <div style="font-size:1.6rem; font-weight:700; color:{score_color};">
                    {r['avg_health_score']:.1f}
                    <span style="font-size:0.8rem; color:#555; font-weight:400;">
                      &nbsp;avg health
                    </span>
                  </div>
                  <div style="font-size:0.82rem; color:#333; margin-top:8px;
                              display:grid; grid-template-columns:1fr 1fr; gap:4px 10px;">
                    <div><b>MTBF:</b> {r['avg_mtbf']:.0f} hrs</div>
                    <div><b>MTTR:</b> {r['avg_mttr']:.2f} hrs</div>
                    <div><b>Avail.:</b> {r['avg_availability']:.1f}%</div>
                    <div><b>Fails:</b> {r['total_failures']:.0f}</div>
                  </div>
                  <div style="margin-top:8px; font-size:0.82rem;">
                    <b>Trend:</b>
                    <span style="color:{slope_color}; font-weight:700;">
                      {slope_sign}{r['avg_slope']:.2f} pts/mo
                    </span>
                    &nbsp;&nbsp;
                    <b>Monthly Cost:</b>
                    <span style="color:#0D1B2A; font-weight:700;">
                      ${r['total_cost']:,.0f}
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Fleet Risk Table ──────────────────────────────────────────────────
        st.markdown("#### Machine Risk Ranking")
        st.caption(
            "Machines sorted by forecast risk. HIGH = forecast score < 40, "
            f"MEDIUM = score < {alert_threshold}, LOW = score >= {alert_threshold}."
        )

        risk_df = generate_fleet_risk_table(
            forecasts, monthly, metrics, alert_threshold=float(alert_threshold))

        # Apply VSM filter to risk table if set
        if filter_vsm  != "All": risk_df = risk_df[risk_df["VSM"]  == filter_vsm]
        if filter_area != "All": risk_df = risk_df[risk_df["Area"] == filter_area]

        def _style_risk(val):
            if val == "HIGH":   return "background:#FDECEA; color:#7B241C; font-weight:700;"
            if val == "MEDIUM": return "background:#FEF9E7; color:#9A7D0A; font-weight:700;"
            return "background:#EAFAF1; color:#1E8449; font-weight:700;"

        def _style_trend(val):
            if val == "Deteriorating": return "color:#E74C3C; font-weight:600;"
            if val == "Improving":     return "color:#2ECC71; font-weight:600;"
            return "color:#888;"

        st.dataframe(
            _r(risk_df).style
                .map(_style_risk,  subset=["Risk Level"])
                .map(_style_trend, subset=["Trend"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score (Current)":  st.column_config.NumberColumn(format="%.2f"),
                "Score (Forecast)": st.column_config.NumberColumn(format="%.2f"),
                "CI Low":           st.column_config.NumberColumn(format="%.2f"),
                "CI High":          st.column_config.NumberColumn(format="%.2f"),
                "Delta":            st.column_config.NumberColumn(format="%+.2f"),
                "VSM":              st.column_config.TextColumn(),
                "Area":             st.column_config.TextColumn(),
                "Machine":          st.column_config.TextColumn(),
                "Trend":            st.column_config.TextColumn(),
                "Risk Level":       st.column_config.TextColumn(),
            },
        )

        st.markdown("---")

        # ── Machine Deep Dive ─────────────────────────────────────────────────
        st.markdown("#### Machine Deep Dive")

        all_machines = sorted(metrics["machine_id"].unique())
        machine_sel  = st.selectbox(
            "Select machine", all_machines, index=0,
            key="deep_dive_machine",
        )

        col_chart, col_insight = st.columns([0.65, 0.35])

        with col_chart:
            fig_fc = plot_machine_forecast(
                monthly, machine_sel, horizon=forecast_horizon)
            st.pyplot(fig_fc, use_container_width=True)

        with col_insight:
            fc_match = [f for f in forecasts if f.machine_id == machine_sel]
            if fc_match:
                fc = fc_match[0]
                insight = generate_machine_insight(fc, monthly)

                score_color = "#2ECC71" if fc.health_score >= 70 \
                    else ("#F39C12" if fc.health_score >= 40 else "#E74C3C")
                risk_lbl = "HIGH" if fc.health_score < 40 \
                    else ("MEDIUM" if fc.health_score < alert_threshold else "LOW")
                risk_bg  = {"HIGH": "#FDECEA", "MEDIUM": "#FEF9E7",
                            "LOW": "#EAFAF1"}[risk_lbl]
                risk_txt = {"HIGH": "#7B241C", "MEDIUM": "#9A7D0A",
                            "LOW": "#1E8449"}[risk_lbl]

                st.markdown(
                    f"""
                    <div style="background:#F4F6F7; border-radius:8px;
                                padding:14px 16px; margin-top:6px;">
                      <div style="font-size:1.0rem; font-weight:700;
                                  color:#0D1B2A; margin-bottom:10px;">
                        {machine_sel} &nbsp;
                        <span style="font-size:0.78rem; color:#555;
                                     font-weight:400;">({fc.vsm} / {fc.area})</span>
                      </div>

                      <div style="display:grid; grid-template-columns:1fr 1fr;
                                  gap:6px 14px; font-size:0.83rem; margin-bottom:10px;">
                        <div><b>Forecast Score</b></div>
                        <div style="color:{score_color}; font-weight:700;">
                          {fc.health_score:.1f}
                        </div>
                        <div><b>80% CI</b></div>
                        <div>{fc.health_lo:.1f} – {fc.health_hi:.1f}</div>
                        <div><b>MTBF fcast</b></div>
                        <div>{fc.mtbf_hrs:.0f} hrs</div>
                        <div><b>MTTR fcast</b></div>
                        <div>{fc.mttr_hrs:.2f} hrs</div>
                        <div><b>Avail. fcast</b></div>
                        <div>{fc.availability:.1f}%</div>
                        <div><b>Fails fcast</b></div>
                        <div>{fc.failures:.1f}</div>
                        <div><b>Cost fcast</b></div>
                        <div style="color:#E74C3C;">${fc.downtime_cost:,.0f}</div>
                        <div><b>Slope</b></div>
                        <div>{fc.health_slope:+.2f} pts/mo
                          ({fc.n_months} months)
                        </div>
                      </div>

                      <div style="background:{risk_bg};
                                  border-radius:6px; padding:6px 10px;
                                  font-size:0.82rem; font-weight:700;
                                  color:{risk_txt}; margin-bottom:10px;">
                        Risk: {risk_lbl}
                        {'  (unreliable: < 6 months of data)' if not fc.reliable else ''}
                      </div>

                      <div style="font-size:0.82rem; color:#444; line-height:1.55;">
                        {insight}
                      </div>

                      <div style="margin-top:10px; padding-top:8px;
                                  border-top:1px solid #DDD;
                                  font-size:0.75rem; color:#888;">
                        Trend signal: <b style="color:{
                            '#27AE60' if fc.r2_adj >= 0.60 else
                            '#F39C12' if fc.r2_adj >= 0.20 else
                            '#E74C3C' if fc.r2_adj >= 0.00 else '#999'
                        };">{
                            'Strong' if fc.r2_adj >= 0.60 else
                            'Moderate' if fc.r2_adj >= 0.20 else
                            'Weak' if fc.r2_adj >= 0.00 else 'Noisy — no clear trend'
                        }</b>
                        &nbsp;&nbsp;·&nbsp;&nbsp;
                        &plusmn;{fc.mae:.1f} pts accuracy
                        &nbsp;&nbsp;·&nbsp;&nbsp;
                        n = {fc.n_months} months
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No forecast available for this machine.")

        # ── Technical methodology notes ───────────────────────────────────────
        st.markdown("---")
        with st.expander("📐 Model Methodology & Technical Notes", expanded=False):
            st.markdown(
                """
                #### Why Linear Regression — and Why Low R² is Expected

                The forecasting engine uses **ordinary least squares linear regression**
                fitted independently on each machine's monthly health score time series.

                In a well-run maintenance operation, health scores **should not follow a
                clean linear trend** — and that is by design, not a model failure.
                Every corrective action, preventive maintenance cycle, or part replacement
                deliberately interrupts the deterioration curve and resets the trajectory.
                A machine showing a perfect downward R²= 0.95 would be the real warning signal:
                it would mean no interventions are happening and the machine is in free fall.

                **Low R² values here are a sign that the maintenance team is doing its job.**
                The model captures the *velocity of deterioration between interventions*,
                not a fixed long-term destiny. If the slope is −3 pts/month, the team has
                a quantified time window to act before crossing a critical threshold —
                regardless of what happened in previous months.

                | Criterion | Justification |
                |---|---|
                | **Intervention-aware context** | Each maintenance action resets the trend. Linear regression measures the current deterioration rate, not a permanent trajectory. |
                | **~16 months of history** | Complex models (Random Forest, LSTM, XGBoost) overfit severely with fewer than 30–50 observations. A linear trend is statistically more honest at this scale. |
                | **Auditability** | Every forecast reduces to a slope and an intercept. A reliability engineer can verify, challenge, or override it without opening a black box. |
                | **Interpretability** | *"This machine is losing 2.3 health points per month"* is a statement a plant director understands and can act on immediately. |
                | **Stability** | No hyperparameters to tune, no risk of data leakage across time windows, no distributional assumptions beyond linearity. |

                ---

                #### Why 80% Confidence Interval — not 95%?

                With 16 data points, a 95% CI would span almost the entire 0–100 health
                score range — visually and operationally useless.
                The **80% interval** is a practical compromise: it communicates genuine
                uncertainty without producing bands so wide they lose meaning.
                A wide band on a specific machine is itself a signal — it means that
                machine's behavior is erratic between interventions, which is exactly
                the kind of insight that should trigger closer monitoring frequency.

                ---

                #### Trend Signal Quality (shown in Machine Deep Dive)

                - **Strong / Moderate** — the machine shows a consistent directional pattern
                  between maintenance events. The slope is a reliable leading indicator.
                - **Weak** — some directional signal exists but interventions are frequent
                  enough to introduce significant noise. Use the slope as a soft indicator only.
                - **Noisy — no clear trend** — the machine's health score fluctuates without
                  a dominant direction. This is often the healthiest operational state:
                  maintenance actions are keeping the machine within a stable band.
                  The ±MAE value still provides a useful accuracy reference for the forecast range.

                ---

                #### How the Model Evolves With More Data

                The model accuracy improves as more months of real operational data accumulate.
                The architecture is designed so the forecasting engine (`src/ml/forecasting.py`)
                can be upgraded without changing any other part of the dashboard —
                the `MachineForecast` dataclass acts as a stable interface between the
                ML layer and the visualization layer.

                | Data available | Recommended model |
                |---|---|
                | **< 18 months** | Linear regression (current) — most stable and honest option |
                | **18–36 months** | **Holt-Winters ETS** — captures level + trend + maintenance cycle seasonality |
                | **36–60 months** | **ARIMA(1,1,0)** — models autocorrelation in residuals between interventions |
                | **60+ months** | **LightGBM with lag features** — captures non-linear degradation patterns and interaction effects between machines |
                """
            )


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TAB 4 — ROOT CAUSE & PRESCRIPTIVE
# ╚══════════════════════════════════════════════════════════════════════════════
with tab4:

    if components is None or components.empty:
        st.warning(
            "Component metrics not found. Click **Refresh Data** in the sidebar "
            "to generate `component_metrics.csv` from the ETL pipeline."
        )
    else:
        # Apply VSM/Area filter to component data
        comp_f = components.copy()
        if filter_vsm  != "All": comp_f = comp_f[comp_f["vsm"]  == filter_vsm]
        if filter_area != "All": comp_f = comp_f[comp_f["area"] == filter_area]

        # ── Generate prescriptive actions ─────────────────────────────────────
        with st.spinner("Running prescriptive engine..."):
            if monthly is not None:
                forecasts_rc = generate_machine_forecasts(monthly, horizon=forecast_horizon)
                fc_df_rc     = forecasts_to_df(forecasts_rc)
            else:
                fc_df_rc = None

            actions     = generate_prescriptive_actions(
                comp_f, forecasts_df=fc_df_rc,
                inaction_months=3)
            actions_df  = actions_to_df(actions) if actions else pd.DataFrame()
            exec_summary = generate_executive_summary(actions, inaction_months=3)
            roi_table    = compute_component_roi_table(comp_f)

        # ── Executive banner ──────────────────────────────────────────────────
        imm   = exec_summary["immediate"]
        total_act = exec_summary["cost_to_act"]
        total_ign = exec_summary["cost_if_ignored"]
        roi_pct   = exec_summary["roi_pct"]
        crit_m    = exec_summary["critical_machines"]

        banner_color = "#7B241C" if imm > 0 else "#1A5276"
        st.markdown(
            f"""
            <div style="
              background: linear-gradient(135deg, {banner_color} 0%, #1A3552 100%);
              border-radius: 10px; padding: 18px 24px; margin-bottom: 18px; color: white;
            ">
              <div style="font-size:1.1rem; font-weight:700; letter-spacing:0.5px;">
                Root Cause & Prescriptive Intelligence
                &nbsp;&mdash;&nbsp; {exec_summary['total_actions']} actions identified
              </div>
              <div style="display:grid; grid-template-columns: repeat(4,1fr);
                          gap:12px; margin-top:12px;">
                <div style="background:rgba(255,255,255,0.12); border-radius:6px; padding:10px;">
                  <div style="font-size:1.5rem; font-weight:800; color:#E74C3C;">{imm}</div>
                  <div style="font-size:0.78rem; color:#CCC;">Immediate actions</div>
                </div>
                <div style="background:rgba(255,255,255,0.12); border-radius:6px; padding:10px;">
                  <div style="font-size:1.5rem; font-weight:800; color:#F39C12;">${total_act:,.0f}</div>
                  <div style="font-size:0.78rem; color:#CCC;">Cost to act (all actions)</div>
                </div>
                <div style="background:rgba(255,255,255,0.12); border-radius:6px; padding:10px;">
                  <div style="font-size:1.5rem; font-weight:800; color:#E84C1F;">${total_ign:,.0f}</div>
                  <div style="font-size:0.78rem; color:#CCC;">Cost if ignored (3 months)</div>
                </div>
                <div style="background:rgba(255,255,255,0.12); border-radius:6px; padding:10px;">
                  <div style="font-size:1.5rem; font-weight:800; color:#2ECC71;">{roi_pct:.0f}%</div>
                  <div style="font-size:0.78rem; color:#CCC;">ROI of preventive plan</div>
                </div>
              </div>
              {f'<div style="margin-top:10px; font-size:0.82rem; color:#F1948A;">Immediate attention: {", ".join(crit_m[:8])}</div>' if crit_m else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Section A: Component Pareto ───────────────────────────────────────
        st.markdown("#### A — Component Cost Pareto (80/20)")
        st.caption(
            "📊 Historical data  ·  "
            "Which subsystems are consuming the most maintenance budget? "
            "Each bar shows the total accumulated downtime cost per component type "
            "(spindle, servo drive, coolant system, etc.). "
            "The line tracks the cumulative percentage — the first 3 or 4 components "
            "that cross the 80% mark are the highest-priority targets for preventive action. "
            "Bar colors indicate failure type: blue = mechanical, red = electrical, "
            "purple = software/automation, green = hydraulic."
        )
        fig_pareto = plot_component_pareto(roi_table, top_n=12)
        st.pyplot(fig_pareto, use_container_width=True)
        st.info(
            "💡 **How to read it:** If the spindle appears first with 40% of total cost, "
            "that single component accounts for nearly half of all production downtime spending. "
            "Focusing preventive efforts there delivers the highest return on investment.",
            icon=None,
        )

        st.markdown("---")

        # ── Section B: Component × Machine Heatmap ───────────────────────────
        st.markdown("#### B — Component × Machine Downtime Heatmap")
        st.caption(
            "📊 Historical data  ·  "
            "Heat matrix: rows = machines, columns = components. "
            "Each cell color represents accumulated downtime hours for that component "
            "on that specific machine — darker = more time stopped. "
            "Use the VSM filter to focus the analysis on a single value stream."
        )
        vsm_rc = st.radio(
            "VSM filter for heatmap",
            ["All", "Alpha", "Beta", "Gamma"],
            horizontal=True, key="rc_vsm_heatmap",
        )
        fig_hmap = plot_component_machine_heatmap(components, vsm_filter=vsm_rc)
        st.pyplot(fig_hmap, use_container_width=True)
        st.info(
            "💡 **How to read it:** If an entire column is dark (e.g. 'coolant_system' "
            "across multiple machines), the problem is systemic across the VS — likely a shared "
            "design flaw, supplier issue, or common procedure. "
            "If only one cell is dark, the failure is isolated to that specific machine "
            "and may have an individual root cause.",
            icon=None,
        )

        st.markdown("---")

        # ── Section C: MTTR by failure type ──────────────────────────────────
        st.markdown("#### C — Avg MTTR by Failure Type & Area")
        st.caption(
            "📊 Historical data  ·  "
            "Average repair time (MTTR) grouped by failure type and production area. "
            "This chart does not show how often a component fails — "
            "it shows how long it takes to get back online once it does. "
            "A high MTTR points to diagnostic bottlenecks, missing spare parts, "
            "or the need for a specialist technician."
        )
        fig_mttr = plot_mttr_by_failure_type(comp_f if not comp_f.empty else components)
        st.pyplot(fig_mttr, use_container_width=True)
        st.info(
            "💡 **How to read it:** Electrical failures tend to have the highest MTTR "
            "due to diagnostic complexity. Software/automation faults can spike if no "
            "automation technician is available on shift. "
            "If Assembly MTTR is higher than Machining for the same failure type, "
            "it may indicate physical access constraints or missing tools in that area.",
            icon=None,
        )

        st.markdown("---")

        # ── Section D: Prescriptive Action Table ─────────────────────────────
        st.markdown("#### D — Prioritized Prescriptive Action Plan")
        st.caption(
            "🔀 Historical data + forecast trend  ·  "
            "Prioritized action plan generated automatically by the prescriptive engine. "
            "Each recommendation combines historical failure frequency, "
            "repeat patterns within 90-day windows, and the health score deterioration slope "
            "from the predictive forecast. "
            "This is not just a prediction — it is a concrete recommendation: "
            "what to do, who does it, which part to order, and how much it costs to act vs. to ignore."
        )
        st.info(
            "💡 **How to read it:** "
            "'Act Cost' = part cost + estimated technician hours. "
            "'Ignore Cost' = 3-month projection if no action is taken (based on historical failure rate). "
            "'ROI' = how much is saved per dollar invested in the preventive action. "
            "Urgency 'Immediate' = requires attention before the next production shift.",
            icon=None,
        )

        if actions_df.empty:
            st.success(
                "No prescriptive triggers found for current filters — "
                "all components within acceptable thresholds."
            )
        else:
            # Urgency filter
            urgency_filter = st.multiselect(
                "Filter by urgency",
                ["Immediate", "This week", "Within 2 weeks", "Next PM cycle"],
                default=["Immediate", "This week", "Within 2 weeks"],
                key="rc_urgency_filter",
            )
            disp = actions_df[actions_df["urgency"].isin(urgency_filter)].copy()

            # Display columns
            cols_show = [
                "urgency", "machine_id", "vsm", "area", "component",
                "failure_type", "technician_type", "action_label",
                "part_replaced", "part_cost_usd", "part_lead_time_days",
                "cost_to_act", "cost_if_ignored", "roi_pct",
                "confidence", "rationale",
            ]
            disp_show = disp[[c for c in cols_show if c in disp.columns]]

            def _style_urgency(val):
                colors = {
                    "Immediate":      "background:#FDECEA; color:#7B241C; font-weight:700;",
                    "This week":      "background:#FDFAE7; color:#7D6608; font-weight:700;",
                    "Within 2 weeks": "background:#FEF9E7; color:#9A7D0A;",
                    "Next PM cycle":  "background:#EAFAF1; color:#1E8449;",
                }
                return colors.get(val, "")

            def _style_conf(val):
                return ("color:#27AE60; font-weight:700;" if val == "High"
                        else "color:#F39C12;" if val == "Medium"
                        else "color:#E74C3C;")

            st.dataframe(
                _r(disp_show).style
                    .map(_style_urgency, subset=["urgency"])
                    .map(_style_conf,    subset=["confidence"]),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "urgency":             st.column_config.TextColumn("Urgency"),
                    "machine_id":          st.column_config.TextColumn("Machine"),
                    "vsm":                 st.column_config.TextColumn("VSM"),
                    "area":                st.column_config.TextColumn("Area"),
                    "component":           st.column_config.TextColumn("Component"),
                    "failure_type":        st.column_config.TextColumn("Failure Type"),
                    "technician_type":     st.column_config.TextColumn("Technician"),
                    "action_label":        st.column_config.TextColumn("Action"),
                    "part_replaced":       st.column_config.TextColumn("Part"),
                    "confidence":          st.column_config.TextColumn("Confidence"),
                    "rationale":           st.column_config.TextColumn("Rationale", width="large"),
                    "part_cost_usd":       st.column_config.NumberColumn("Part Cost ($)",    format="$%.2f"),
                    "cost_to_act":         st.column_config.NumberColumn("Act Cost ($)",     format="$%.2f"),
                    "cost_if_ignored":     st.column_config.NumberColumn("Ignore Cost ($)",  format="$%.2f"),
                    "roi_pct":             st.column_config.NumberColumn("ROI (%)",          format="%.2f"),
                    "part_lead_time_days": st.column_config.NumberColumn("Lead (days)",      format="%d"),
                },
            )

        # ── Section E: Cost vs Urgency visual ────────────────────────────────
        st.markdown("---")
        st.markdown("#### E — Cost to Act vs Cost if Ignored")
        st.caption(
            "📈 3-month projection  ·  "
            "Side-by-side comparison per action: left bar = cost of intervening now "
            "(part + labor), right bar = projected cost of doing nothing "
            "over the next 3 months. "
            "The inaction cost is calculated as historical failure rate x average cost per event "
            "x 3 months — this is not an ML model, it is risk arithmetic grounded in real data."
        )
        if not actions_df.empty:
            fig_urg = plot_prescriptive_urgency_chart(actions_df)
            st.pyplot(fig_urg, use_container_width=True)
            st.info(
                "💡 **How to read it:** When the right bar (ignore) is much taller than "
                "the left (act), the ROI of the preventive action is high — worth the investment. "
                "When both bars are similar in height, urgency is lower and the action "
                "can be scheduled in the next planned maintenance cycle.",
                icon=None,
            )

        # ── Section F: VS ROI table ────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### F — Component ROI Summary (Value Stream-wide)")
        st.caption(
            "📊 Historical data  ·  "
            "Consolidated view of which components generate the most maintenance cost "
            "across all Value Streams (Alpha, Beta, Gamma) and all machines combined. "
            "Unlike the Pareto chart (section A), this table also shows how many machines "
            "are affected per component and whether the pattern repeats within 90-day windows. "
            "Use this to negotiate maintenance contracts, define which spare parts to stock "
            "per VS, and detect systemic failures that cross production lines."
        )
        if not roi_table.empty:
            def _style_cumul(val):
                if val <= 50:  return "color:#E74C3C; font-weight:700;"
                if val <= 80:  return "color:#F39C12; font-weight:600;"
                return "color:#27AE60;"

            st.dataframe(
                _r(roi_table).style.map(_style_cumul, subset=["cumul_pct"]),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "component":          st.column_config.TextColumn("Component"),
                    "failure_type":       st.column_config.TextColumn("Failure Type"),
                    "total_failures":     st.column_config.NumberColumn("Failures",         format="%d"),
                    "machines_affected":  st.column_config.NumberColumn("Machines",         format="%d"),
                    "repeat_count":       st.column_config.NumberColumn("Repeat (90d)",     format="%d"),
                    "total_downtime_hrs": st.column_config.NumberColumn("Downtime (hrs)",   format="%.2f"),
                    "total_cost":         st.column_config.NumberColumn("Total Cost ($)",   format="$%.2f"),
                    "avg_mttr":           st.column_config.NumberColumn("Avg MTTR (hrs)",   format="%.2f"),
                    "pct_cost":           st.column_config.NumberColumn("% of Total",       format="%.2f"),
                    "cumul_pct":          st.column_config.NumberColumn("Cumul. %",         format="%.2f"),
                },
            )
            st.info(
                "💡 **How to read it:** '% of Total' shows what share of the total maintenance "
                "cost that component represents across the entire operation. "
                "'Cumul. %' in red means the component falls within the Pareto 80% threshold — "
                "highest priority. "
                "'Repeat (90d)' counts how many machines had that same component fail "
                "3 or more times within any 90-day window — a signal of a recurring failure "
                "that requires a structural fix, not just a reactive repair.",
                icon=None,
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding:20px 0 6px; color:#aaa; font-size:0.78rem;">
      LozanoLsa &nbsp;|&nbsp; Value Stream Health Intelligence v2.0
      &nbsp;|&nbsp; Bridging Operational Excellence &amp; Predictive Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)
