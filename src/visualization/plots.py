"""
Reusable KPI plotting helpers for the dashboard.
All numeric values are rounded to 2 decimal places at the source.
"""
import pandas as pd


def _round2(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number").columns
    return df.assign(**{c: df[c].round(2) for c in num})


def vsm_summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    grp = metrics.groupby("vsm").agg(
        avg_mtbf=("mtbf_hrs", "mean"),
        avg_mttr=("mttr_hrs", "mean"),
        avg_availability=("availability_pct", "mean"),
        failures_30d=("failures_30d", "sum"),
        monthly_downtime_cost=("total_monthly_downtime_cost", "sum"),
    ).round(2)
    grp["monthly_downtime_cost"] = grp["monthly_downtime_cost"].apply(
        lambda v: f"${v:,.2f}")
    grp = grp.rename(columns={
        "avg_mtbf":              "Avg MTBF (hrs)",
        "avg_mttr":              "Avg MTTR (hrs)",
        "avg_availability":      "Avg Availability (%)",
        "failures_30d":          "Failures (30d)",
        "monthly_downtime_cost": "Monthly Downtime Cost",
    })
    return grp.reset_index().rename(columns={"vsm": "VSM"})


def top_critical(metrics: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    crit = (metrics[metrics["health_status"] == "Critical"]
            .sort_values("health_score")
            .head(n)[["machine_id", "vsm", "area", "health_score",
                       "mtbf_hrs", "mttr_hrs", "failures_30d"]])
    crit = _round2(crit)
    return crit.rename(columns={
        "machine_id":   "Machine",
        "vsm":          "VSM",
        "area":         "Area",
        "health_score": "Score",
        "mtbf_hrs":     "MTBF (hrs)",
        "mttr_hrs":     "MTTR (hrs)",
        "failures_30d": "Fails (30d)",
    })


def top_healthy(metrics: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    ok = (metrics[metrics["health_status"] == "Healthy"]
          .sort_values("health_score", ascending=False)
          .head(n)[["machine_id", "vsm", "area", "health_score",
                    "mtbf_hrs", "mttr_hrs", "availability_pct", "failures_30d"]])
    ok = _round2(ok)
    return ok.rename(columns={
        "machine_id":       "Machine",
        "vsm":              "VSM",
        "area":             "Area",
        "health_score":     "Score",
        "mtbf_hrs":         "MTBF (hrs)",
        "mttr_hrs":         "MTTR (hrs)",
        "availability_pct": "Availability (%)",
        "failures_30d":     "Fails (30d)",
    })
