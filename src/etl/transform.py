import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.utils.helpers import clamp

log = get_logger(__name__)

LOOKBACK_DAYS = 30


def compute_mtbf_metrics(equipment: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates MTBF, MTTR, 30-day failure count, availability, health score,
    and monthly downtime cost per machine.
    """
    cutoff = failures["failure_date"].max()
    window_start = cutoff - timedelta(days=LOOKBACK_DAYS)
    recent = failures[failures["failure_date"] >= window_start]

    records = []
    for _, eq in equipment.iterrows():
        mid = eq["machine_id"]
        mfails = failures[failures["machine_id"] == mid].copy()
        mfails_recent = recent[recent["machine_id"] == mid]

        # MTBF: total observed period / number of failures
        if len(mfails) >= 2:
            total_days = (mfails["failure_date"].max() - mfails["failure_date"].min()).days
            total_hrs = max(total_days * 24, 1)
            mtbf = round(total_hrs / len(mfails), 1)
        elif len(mfails) == 1:
            mtbf = 720.0  # one failure in period → assume ~1 month between failures
        else:
            mtbf = 999.9  # no failures → very reliable

        # MTTR: mean repair time from all records
        if len(mfails) > 0:
            mttr = round(mfails["repair_hrs"].mean(), 2)
        else:
            mttr = 0.5

        failures_30d = int(len(mfails_recent))
        total_downtime_30d = float(mfails_recent["downtime_hrs"].sum())

        # Availability: (available hours - downtime) / available hours
        available_hrs = LOOKBACK_DAYS * 24
        avail_pct = round(max(0.0, (available_hrs - total_downtime_30d) / available_hrs) * 100, 2)

        downtime_cost_hr = float(eq["downtime_cost_per_hr"])
        monthly_downtime_cost = round(total_downtime_30d * downtime_cost_hr, 2)

        records.append({
            "machine_id":               mid,
            "vsm":                      eq["vsm"],
            "area":                     eq["area"],
            "machine_type":             eq["machine_type"],
            "mtbf_hrs":                 mtbf,
            "mttr_hrs":                 mttr,
            "failures_30d":             failures_30d,
            "availability_pct":         avail_pct,
            "downtime_cost_per_hr":     downtime_cost_hr,
            "total_monthly_downtime_cost": monthly_downtime_cost,
        })

    df = pd.DataFrame(records)

    # Health score (fleet-normalized)
    max_mtbf = df["mtbf_hrs"].replace(999.9, np.nan).max()
    max_fails = df["failures_30d"].max() if df["failures_30d"].max() > 0 else 1

    def _score(row):
        norm_mtbf  = min(row["mtbf_hrs"], max_mtbf) / max_mtbf if max_mtbf else 1.0
        norm_avail = row["availability_pct"] / 100.0
        norm_fail  = 1.0 - (row["failures_30d"] / max_fails)
        raw = norm_mtbf * 0.5 + norm_avail * 0.3 + norm_fail * 0.2
        return round(clamp(raw * 100), 1)

    df["health_score"] = df.apply(_score, axis=1)

    def _status(score):
        if score >= 70:
            return "Healthy"
        elif score >= 40:
            return "Monitor"
        return "Critical"

    df["health_status"] = df["health_score"].apply(_status)

    # Round all numeric columns to 2 decimal places
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(2)

    log.info(f"  Health distribution: "
             f"Healthy={( df['health_status']=='Healthy').sum()}  "
             f"Monitor={( df['health_status']=='Monitor').sum()}  "
             f"Critical={(df['health_status']=='Critical').sum()}")
    return df


def clean_failures(failures: pd.DataFrame) -> pd.DataFrame:
    df = failures.dropna(subset=["machine_id", "failure_date"]).copy()
    df["downtime_hrs"] = df["downtime_hrs"].clip(lower=0)
    df["repair_hrs"]   = df["repair_hrs"].clip(lower=0)
    return df


def compute_monthly_metrics(equipment: pd.DataFrame,
                             failures: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates failure and availability data by machine × month.
    Produces the time-series used by the ML forecasting layer.
    """
    MAX_MTBF_CAP = 744.0   # max hrs in a 31-day month

    failures = failures.copy()
    failures["year_month"] = failures["failure_date"].dt.to_period("M")

    min_month = failures["failure_date"].dt.to_period("M").min()
    max_month = failures["failure_date"].dt.to_period("M").max()
    all_months = pd.period_range(min_month, max_month, freq="M")

    records = []
    for _, eq in equipment.iterrows():
        mid    = eq["machine_id"]
        mfails = failures[failures["machine_id"] == mid]

        for month in all_months:
            mf            = mfails[mfails["year_month"] == month]
            days          = month.days_in_month
            available_hrs = float(days * 24)
            fail_count    = len(mf)
            downtime      = float(mf["downtime_hrs"].sum())
            mttr          = float(mf["repair_hrs"].mean()) if fail_count > 0 else 0.0
            operating     = max(0.0, available_hrs - downtime)
            mtbf          = min(operating / fail_count, MAX_MTBF_CAP) if fail_count > 0 else available_hrs
            avail         = max(0.0, (available_hrs - downtime) / available_hrs * 100)
            cost          = round(downtime * float(eq["downtime_cost_per_hr"]), 2)

            records.append({
                "year_month":            str(month),
                "machine_id":            mid,
                "vsm":                   eq["vsm"],
                "area":                  eq["area"],
                "machine_type":          eq["machine_type"],
                "failures_count":        fail_count,
                "downtime_hrs":          round(downtime, 2),
                "mtbf_hrs":              round(mtbf, 1),
                "mttr_hrs":              round(mttr, 2),
                "availability_pct":      round(avail, 2),
                "downtime_cost_per_hr":  float(eq["downtime_cost_per_hr"]),
                "monthly_downtime_cost": cost,
            })

    df = pd.DataFrame(records)

    # Consistent health score normalization across all months
    max_fails = max(int(df["failures_count"].max()), 1)

    def _score(row):
        nm = min(row["mtbf_hrs"], MAX_MTBF_CAP) / MAX_MTBF_CAP
        na = row["availability_pct"] / 100.0
        nf = 1.0 - (row["failures_count"] / max_fails)
        return round(clamp((nm * 0.5 + na * 0.3 + nf * 0.2) * 100), 1)

    df["health_score"] = df.apply(_score, axis=1)
    df["health_status"] = df["health_score"].apply(
        lambda s: "Healthy" if s >= 70 else "Monitor" if s >= 40 else "Critical"
    )

    # Round all numeric columns to 2 decimal places
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(2)

    log.info(f"  Monthly metrics: {len(df)} rows "
             f"({df['machine_id'].nunique()} machines x {df['year_month'].nunique()} months)")
    return df


def compute_component_metrics(equipment: pd.DataFrame,
                               failures: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates failure data by component / failure_type / machine.
    Powers the Root Cause & Prescriptive tab.
    Returns one row per (machine_id, component) combination.
    """
    required = {"component", "failure_type", "technician_type",
                "part_replaced", "part_cost_usd", "part_lead_time_days",
                "time_to_diagnose_hrs"}
    if not required.issubset(set(failures.columns)):
        log.warning("failures.csv missing component fields — skipping component metrics")
        return pd.DataFrame()

    eq_map = equipment.set_index("machine_id")[
        ["vsm", "area", "machine_type", "downtime_cost_per_hr"]
    ].to_dict("index")

    records = []
    for (mid, comp), grp in failures.groupby(["machine_id", "component"]):
        eq = eq_map.get(mid, {})
        n          = len(grp)
        downtime   = float(grp["downtime_hrs"].sum())
        repair     = float(grp["repair_hrs"].sum())
        mttr_comp  = round(repair / n, 2) if n > 0 else 0.0
        avg_diag   = round(float(grp["time_to_diagnose_hrs"].mean()), 2)
        avg_cost   = round(float(grp["part_cost_usd"].mean()), 2)
        lead_time  = int(grp["part_lead_time_days"].median())
        cost_hr    = float(eq.get("downtime_cost_per_hr", 100))
        total_cost = round(downtime * cost_hr, 2)

        # Dominant failure_type / technician for this component
        ftype = grp["failure_type"].mode().iloc[0]
        tech  = grp["technician_type"].mode().iloc[0]
        part  = grp["part_replaced"].mode().iloc[0]

        # Repeat-failure flag: same component failed 3+ times in any 90-day window
        dates     = grp["failure_date"].sort_values().reset_index(drop=True)
        repeat_90 = False
        for i in range(len(dates)):
            window = dates[(dates >= dates[i]) & (dates <= dates[i] + pd.Timedelta(days=90))]
            if len(window) >= 3:
                repeat_90 = True
                break

        records.append({
            "machine_id":            mid,
            "vsm":                   eq.get("vsm", ""),
            "area":                  eq.get("area", ""),
            "machine_type":          eq.get("machine_type", ""),
            "component":             comp,
            "failure_type":          ftype,
            "technician_type":       tech,
            "part_replaced":         part,
            "failure_count":         n,
            "total_downtime_hrs":    round(downtime, 2),
            "total_repair_hrs":      round(repair, 2),
            "mttr_component_hrs":    mttr_comp,
            "avg_diagnose_hrs":      avg_diag,
            "avg_part_cost_usd":     avg_cost,
            "part_lead_time_days":   lead_time,
            "total_downtime_cost":   total_cost,
            "repeat_failure_90d":    repeat_90,
        })

    df = pd.DataFrame(records)
    # Pareto rank: component contribution to total downtime cost
    total = df["total_downtime_cost"].sum()
    df = df.sort_values("total_downtime_cost", ascending=False).reset_index(drop=True)
    df["pct_of_total_cost"] = (df["total_downtime_cost"] / total * 100).round(1)
    df["cumulative_pct"]    = df["pct_of_total_cost"].cumsum().round(1)

    # Round all numeric columns to 2 decimal places
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(2)

    log.info(f"  Component metrics: {len(df)} rows "
             f"({df['component'].nunique()} unique components, "
             f"{df['machine_id'].nunique()} machines)")
    return df
