"""
Natural-language explainer for machine health forecasts.
Converts statistical trend signals into executive-readable insights.
"""
import numpy as np
import pandas as pd
from src.ml.forecasting import MachineForecast


def _trend_label(slope: float) -> str:
    if slope < -2.0:  return "deteriorating rapidly"
    if slope < -0.5:  return "in slow decline"
    if slope >  1.5:  return "recovering"
    if slope >  0.3:  return "improving slightly"
    return "stable"


def generate_machine_insight(fc: MachineForecast,
                              monthly_df: pd.DataFrame) -> str:
    """
    Returns a 2-3 sentence insight string for one machine forecast.
    """
    mid   = fc.machine_id
    grp   = monthly_df[monthly_df["machine_id"] == mid].sort_values("year_month")
    n     = len(grp)
    parts = []

    # ── Trend sentence ────────────────────────────────────────────────────────
    trend = _trend_label(fc.health_slope)
    if n >= 3:
        parts.append(
            f"{mid} health score is {trend} "
            f"({fc.health_slope:+.1f} pts/month over {n} months)."
        )
    else:
        parts.append(f"{mid} has limited history ({n} months) — forecast carries higher uncertainty.")

    # ── Failure spike ─────────────────────────────────────────────────────────
    if n >= 3:
        recent_avg  = float(grp["failures_count"].iloc[-2:].mean())
        baseline    = float(grp["failures_count"].mean())
        if baseline > 0 and recent_avg > baseline * 1.6:
            parts.append(
                f"Failure frequency spiked in recent months "
                f"({recent_avg:.1f} vs avg {baseline:.1f}/month)."
            )

    # ── MTTR trend ───────────────────────────────────────────────────────────
    if n >= 4:
        mttr_vals = grp["mttr_hrs"].values
        mttr_slope = float(np.polyfit(np.arange(n), mttr_vals, 1)[0])
        if mttr_slope > 0.2:
            parts.append(
                f"Repair times trending upward (+{mttr_slope:.2f} hrs/month) "
                f"— possible component wear or parts availability issue."
            )

    # ── Forward-looking risk sentence ────────────────────────────────────────
    score = fc.health_score
    if score < 40:
        months_at_risk = 0
        parts.append(
            f"Forecast score {score:.0f} — CRITICAL zone. "
            f"Preventive maintenance should be scheduled immediately."
        )
    elif score < 55:
        parts.append(
            f"Forecast score {score:.0f} — approaching Critical threshold. "
            f"Recommend inspection within 2 weeks."
        )
    elif score < 70:
        parts.append(
            f"Forecast score {score:.0f} — Monitor zone. "
            f"Continue tracking; no urgent action required."
        )
    else:
        parts.append(
            f"Forecast score {score:.0f} — Healthy. "
            f"No action required based on current trend."
        )

    return " ".join(parts)


def generate_fleet_risk_table(forecasts: list[MachineForecast],
                               monthly_df: pd.DataFrame,
                               current_metrics: pd.DataFrame,
                               alert_threshold: float = 55.0) -> pd.DataFrame:
    """
    Returns a risk-ranked table comparing current vs forecast health scores.
    """
    current_map = dict(zip(current_metrics["machine_id"],
                           current_metrics["health_score"]))
    rows = []
    for fc in forecasts:
        current = current_map.get(fc.machine_id, 50.0)
        delta   = fc.health_score - current
        if delta < -3:     trend_icon = "Deteriorating"
        elif delta > 3:    trend_icon = "Improving"
        else:              trend_icon = "Stable"

        risk = "HIGH"   if fc.health_score < 40 else \
               "MEDIUM" if fc.health_score < alert_threshold else "LOW"

        rows.append({
            "Machine":           fc.machine_id,
            "VSM":               fc.vsm,
            "Area":              fc.area,
            "Score (Current)":   current,
            "Score (Forecast)":  fc.health_score,
            "CI Low":            fc.health_lo,
            "CI High":           fc.health_hi,
            "Delta":             round(delta, 1),
            "Trend":             trend_icon,
            "Risk Level":        risk,
        })

    df = pd.DataFrame(rows)
    # Sort: HIGH risk first, then by forecast score ascending
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["_sort"] = df["Risk Level"].map(risk_order)
    df = df.sort_values(["_sort", "Score (Forecast)"]).drop(columns="_sort")
    return df.reset_index(drop=True)
