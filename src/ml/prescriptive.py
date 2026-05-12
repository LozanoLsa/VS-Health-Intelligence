"""
Prescriptive Analytics Engine
Converts component-level failure intelligence into prioritized, costed action plans.

Logic tiers:
  1. Rule-based triggers  (repeat failures, lead-time risk, MTTR spikes)
  2. Trend signal         (health slope from forecasting layer)
  3. Cost-benefit scoring (cost of action vs projected cost of inaction)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Action type catalogue ─────────────────────────────────────────────────────
ACTION_LABELS = {
    "replace":   "Replace component",
    "inspect":   "Inspect & assess",
    "pm":        "Schedule preventive maintenance",
    "order":     "Order spare part now",
    "calibrate": "Recalibrate / re-tune",
    "escalate":  "Escalate to specialist",
}

URGENCY_RANK = {"Immediate": 0, "This week": 1, "Within 2 weeks": 2,
                "Next PM cycle": 3}


@dataclass
class PrescriptiveAction:
    machine_id:          str
    vsm:                 str
    area:                str
    component:           str
    failure_type:        str
    technician_type:     str
    action_type:         str          # key in ACTION_LABELS
    action_label:        str
    urgency:             str
    rationale:           str
    part_replaced:       str
    part_cost_usd:       float
    part_lead_time_days: int
    cost_to_act:         float        # part cost + estimated labour
    cost_if_ignored:     float        # projected downtime cost in 3 months
    roi_pct:             float        # (cost_if_ignored - cost_to_act) / cost_to_act * 100
    confidence:          str          # High / Medium / Low
    priority_score:      float        # lower = more urgent (used for sorting)


# ── Labour cost estimates per technician type (USD/hr) ───────────────────────
LABOUR_RATE = {"mechanical": 65, "electrical": 85,
               "automation": 105, "hydraulic": 75}

# Estimated intervention time (hrs) by action type
INTERVENTION_HRS = {"replace": 4.0, "inspect": 1.5, "pm": 2.5,
                    "order": 0.0, "calibrate": 2.0, "escalate": 1.0}


def _labour_cost(tech: str, action: str) -> float:
    return LABOUR_RATE.get(tech, 75) * INTERVENTION_HRS.get(action, 2.0)


def _confidence(n_failures: int, repeat_90: bool, has_trend: bool) -> str:
    score = 0
    if n_failures >= 5:   score += 2
    elif n_failures >= 3: score += 1
    if repeat_90:         score += 2
    if has_trend:         score += 1
    if score >= 4: return "High"
    if score >= 2: return "Medium"
    return "Low"


def _project_inaction_cost(row: pd.Series, months: int = 3,
                            data_months: int = 16) -> float:
    """
    Estimate cost of doing nothing for `months` months.
    Uses historical failure rate × downtime cost per failure.
    data_months: the actual number of months covered by the input data window
                 (16 for all history, 3/6/12 for rolling windows).
    """
    n      = max(row["failure_count"], 1)
    rate_mo = n / max(data_months, 1)
    projected_fails = rate_mo * months
    avg_downtime_per_fail = (row["total_downtime_hrs"] / n) if n > 0 else 1.0
    cost_per_fail = avg_downtime_per_fail * (row["total_downtime_cost"] / max(row["total_downtime_hrs"], 0.1))
    return round(projected_fails * cost_per_fail, 2)


def generate_prescriptive_actions(
        component_df: pd.DataFrame,
        forecasts_df: pd.DataFrame | None = None,
        inaction_months: int = 3,
        data_months: int = 16,
) -> list[PrescriptiveAction]:
    """
    Generates a prioritized list of PrescriptiveAction for the full machine fleet.

    Triggers (any one is sufficient):
    - repeat_failure_90d == True
    - failure_count >= 3 in the historical window
    - part_lead_time_days >= 7  AND  failure_count >= 2
    - failure_type == 'software' (automation risk)
    - mttr_component_hrs > Value Stream 75th percentile  (slow-to-repair bottleneck)
    - health_slope < -0.5  (from forecasts_df — deteriorating trend)

    data_months: number of months the component_df covers (affects inaction cost projection).
    """
    if component_df.empty:
        return []

    # Merge forecast slope if available
    slope_map: dict[str, float] = {}
    if forecasts_df is not None and "machine_id" in forecasts_df.columns:
        slope_map = dict(zip(forecasts_df["machine_id"],
                             forecasts_df.get("health_slope", pd.Series(dtype=float))))

    # Fleet-relative thresholds (percentile-based so they adapt to any dataset)
    p75_mttr    = float(component_df["mttr_component_hrs"].quantile(0.75))
    p90_fails   = float(component_df["failure_count"].quantile(0.90))
    p75_fails   = float(component_df["failure_count"].quantile(0.75))
    p90_cost    = float(component_df["total_downtime_cost"].quantile(0.90))
    p75_cost    = float(component_df["total_downtime_cost"].quantile(0.75))
    actions: list[PrescriptiveAction] = []

    for _, row in component_df.iterrows():
        mid        = str(row["machine_id"])
        comp       = str(row["component"])
        ftype      = str(row["failure_type"])
        tech       = str(row["technician_type"])
        n_fails    = int(row["failure_count"])
        repeat_90  = bool(row["repeat_failure_90d"])
        mttr_comp  = float(row["mttr_component_hrs"])
        lead_time  = int(row["part_lead_time_days"])
        slope      = slope_map.get(mid, 0.0)
        has_trend  = slope < -0.5
        total_cost = float(row["total_downtime_cost"])

        # Relative flags (vs fleet)
        high_freq  = n_fails  >= p90_fails   # top 10% failure frequency
        mod_freq   = n_fails  >= p75_fails   # top 25% failure frequency
        high_cost  = total_cost >= p90_cost  # top 10% cost impact
        mod_cost   = total_cost >= p75_cost  # top 25% cost impact

        # ── Determine action type & urgency ───────────────────────────────────
        action, urgency, rationale = _select_action(
            ftype, n_fails, repeat_90, mttr_comp, p75_mttr,
            lead_time, slope, row,
            high_freq=high_freq, mod_freq=mod_freq,
            high_cost=high_cost, mod_cost=mod_cost)

        # Skip components with no clear trigger
        if action is None:
            continue

        part_cost   = float(row["avg_part_cost_usd"])
        total_cost  = float(row["total_downtime_cost"])
        labour      = _labour_cost(tech, action)
        cost_to_act = round(part_cost + labour, 2)
        cost_ignore = _project_inaction_cost(row, inaction_months, data_months)
        roi         = round((cost_ignore - cost_to_act) / max(cost_to_act, 1) * 100, 1)

        conf = _confidence(n_fails, repeat_90, has_trend)

        # Priority score: lower = more urgent
        urgency_w   = URGENCY_RANK.get(urgency, 3)
        roi_w       = max(0.0, 100 - roi) / 100
        priority    = round(urgency_w + roi_w + (0 if repeat_90 else 0.5), 3)

        actions.append(PrescriptiveAction(
            machine_id          = mid,
            vsm                 = str(row["vsm"]),
            area                = str(row["area"]),
            component           = comp,
            failure_type        = ftype,
            technician_type     = tech,
            action_type         = action,
            action_label        = ACTION_LABELS[action],
            urgency             = urgency,
            rationale           = rationale,
            part_replaced       = str(row["part_replaced"]),
            part_cost_usd       = part_cost,
            part_lead_time_days = lead_time,
            cost_to_act         = cost_to_act,
            cost_if_ignored     = cost_ignore,
            roi_pct             = roi,
            confidence          = conf,
            priority_score      = priority,
        ))

    actions.sort(key=lambda a: (URGENCY_RANK.get(a.urgency, 9), -a.roi_pct))
    log.info(f"Prescriptive actions generated: {len(actions)}")
    return actions


def _select_action(ftype, n_fails, repeat_90, mttr_comp, p75_mttr,
                   lead_time, slope, row,
                   high_freq=False, mod_freq=False,
                   high_cost=False, mod_cost=False) -> tuple[str | None, str, str]:
    """
    Returns (action_key, urgency, rationale) or (None, '', '') if no trigger.

    Uses fleet-relative flags (high_freq = top 10%, mod_freq = top 25%)
    so thresholds self-calibrate to any dataset size.
    Only the top ~25% of components by combined signals surface as actions.
    """
    total_cost = float(row.get("total_downtime_cost", 0))

    # ── Immediate: top-10% frequency + top-10% cost + declining trend ─────────
    if high_freq and high_cost and slope < -0.5:
        return ("replace", "Immediate",
                f"Component in top 10% of failure frequency ({n_fails} events) AND "
                f"cost impact (${total_cost:,.0f}) with declining health trend "
                f"({slope:+.2f} pts/mo). Replacement is more cost-effective than "
                "continued reactive repair.")

    # ── Immediate: top-10% frequency + repeat pattern ─────────────────────────
    if high_freq and repeat_90 and high_cost:
        return ("replace", "Immediate",
                f"Top-frequency component ({n_fails} failures) with confirmed repeat "
                f"pattern and ${total_cost:,.0f} cumulative cost. "
                "End-of-life failure mode — schedule replacement this sprint.")

    # ── Immediate: lead-time risk + high cost + declining ─────────────────────
    if lead_time >= 12 and high_cost and slope < -0.8:
        return ("order", "Immediate",
                f"Part lead time {lead_time} days, ${total_cost:,.0f} historical cost, "
                f"health trend {slope:+.2f} pts/mo. "
                "Stockout risk is high — order spare part immediately.")

    # ── Immediate: software + top frequency ───────────────────────────────────
    if ftype == "software" and high_freq and high_cost:
        return ("calibrate", "Immediate",
                f"High-frequency software/calibration fault ({n_fails} events, "
                f"${total_cost:,.0f} cost). Sensor drift or firmware regression likely.")

    # ── This week: top-25% + declining + repeat ───────────────────────────────
    if mod_freq and mod_cost and slope < -0.5 and repeat_90:
        return ("replace", "This week",
                f"Recurring failure pattern ({n_fails} events, ${total_cost:,.0f}) "
                f"with health deteriorating at {slope:+.2f} pts/mo. "
                "Replace during next planned production window.")

    # ── This week: lead-time pre-stock ────────────────────────────────────────
    if lead_time >= 10 and mod_cost and slope < -0.3:
        return ("order", "This week",
                f"{lead_time}-day lead time + ${total_cost:,.0f} impact + "
                f"declining trend ({slope:+.2f} pts/mo). "
                "Pre-stock spare to avoid extended downtime on next failure.")

    # ── Within 2 weeks: MTTR outlier ──────────────────────────────────────────
    if mttr_comp > p75_mttr * 2.0 and mod_cost:
        return ("escalate", "Within 2 weeks",
                f"Repair time {mttr_comp:.1f} hrs (VS p75 = {p75_mttr:.1f} hrs). "
                "Diagnostic bottleneck — specialist review to streamline repair process.")

    # ── Within 2 weeks: health slope + moderate cost ──────────────────────────
    if slope < -1.5 and mod_cost:
        return ("inspect", "Within 2 weeks",
                f"Machine health declining {slope:+.2f} pts/mo. "
                f"Component cost impact ${total_cost:,.0f}. "
                "Pre-run inspection recommended before next production cycle.")

    # ── Next PM cycle: high-cost moderate signals ──────────────────────────────
    if high_cost and not mod_freq and not repeat_90:
        return ("pm", "Next PM cycle",
                f"High cost impact (${total_cost:,.0f}) with moderate failure frequency. "
                "No urgent pattern detected — add to next scheduled PM checklist.")

    if ftype == "electrical" and mod_cost and mod_freq:
        return ("inspect", "Next PM cycle",
                f"Electrical component: {n_fails} failures, ${total_cost:,.0f} cost. "
                "Thermographic scan and insulation resistance test recommended at next PM.")

    return (None, "", "")


def actions_to_df(actions: list[PrescriptiveAction]) -> pd.DataFrame:
    return pd.DataFrame([vars(a) for a in actions])


def generate_executive_summary(actions: list[PrescriptiveAction],
                                inaction_months: int = 3) -> dict:
    """
    Returns a dict with top-line numbers for the executive banner.
    """
    if not actions:
        return {"total_actions": 0, "immediate": 0, "cost_to_act": 0,
                "cost_if_ignored": 0, "roi_pct": 0, "critical_machines": []}

    df = actions_to_df(actions)
    immediate_n    = int((df["urgency"] == "Immediate").sum())
    cost_to_act    = round(df["cost_to_act"].sum(), 0)
    cost_if_ignored = round(df["cost_if_ignored"].sum(), 0)
    roi            = round((cost_if_ignored - cost_to_act) / max(cost_to_act, 1) * 100, 1)
    crit_machines  = list(df[df["urgency"] == "Immediate"]["machine_id"].unique())

    return {
        "total_actions":    len(actions),
        "immediate":        immediate_n,
        "cost_to_act":      cost_to_act,
        "cost_if_ignored":  cost_if_ignored,
        "roi_pct":          roi,
        "critical_machines": crit_machines,
        "inaction_months":  inaction_months,
    }


def compute_component_roi_table(component_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fleet-wide Pareto: which components drive the most cost across all machines.
    """
    if component_df.empty:
        return pd.DataFrame()

    grp = (
        component_df.groupby(["component", "failure_type"])
        .agg(
            total_failures   =("failure_count",       "sum"),
            total_downtime_hrs=("total_downtime_hrs",  "sum"),
            total_cost       =("total_downtime_cost",  "sum"),
            avg_mttr         =("mttr_component_hrs",   "mean"),
            machines_affected=("machine_id",           "nunique"),
            repeat_count     =("repeat_failure_90d",   "sum"),
        )
        .reset_index()
        .sort_values("total_cost", ascending=False)
    )
    total_cost = grp["total_cost"].sum()
    grp["pct_cost"]   = (grp["total_cost"] / total_cost * 100).round(1)
    grp["cumul_pct"]  = grp["pct_cost"].cumsum().round(1)
    grp["avg_mttr"]   = grp["avg_mttr"].round(2)
    grp["total_cost"] = grp["total_cost"].round(0)
    return grp.reset_index(drop=True)
