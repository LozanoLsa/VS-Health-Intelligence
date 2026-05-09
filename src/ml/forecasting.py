"""
ML Forecasting Layer — linear trend models with prediction intervals.

For each machine we fit a LinearRegression on its monthly time series
and project forward 1-3 months. Confidence intervals are derived from
the residual standard error (approx. 80% prediction interval).

This is intentionally interpretable: every forecast has a slope, an
intercept, and a residual error that a reliability engineer can audit.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from src.utils.helpers import clamp
from src.utils.logger import get_logger

log = get_logger(__name__)

METRICS = ["health_score", "mtbf_hrs", "mttr_hrs", "availability_pct",
           "failures_count", "monthly_downtime_cost"]

T_80 = 1.282   # ~80% prediction interval z-score


@dataclass
class MachineForecast:
    machine_id:   str
    vsm:          str
    area:         str
    horizon:      int
    # Point forecasts
    health_score: float = 0.0
    health_lo:    float = 0.0
    health_hi:    float = 0.0
    mtbf_hrs:     float = 0.0
    mttr_hrs:     float = 0.0
    availability: float = 0.0
    failures:     float = 0.0
    downtime_cost:float = 0.0
    # Trend metadata
    health_slope: float = 0.0     # pts / month
    n_months:     int   = 0
    reliable:     bool  = False   # True if >= 6 months of data
    # Model fit quality (on historical health score series)
    r2_adj:       float = 0.0     # Adjusted R² — how well trend fits history
    mae:          float = 0.0     # Mean Absolute Error in health score points


def _forecast_series(values: np.ndarray, horizon: int = 1
                     ) -> tuple[float, float, float]:
    """
    Fit a linear trend and return (forecast, lower_80, upper_80).
    Falls back to mean ± 1.5*std for series with < 3 points.
    """
    n = len(values)
    if n < 3:
        mu  = float(np.mean(values))
        sig = float(np.std(values)) if n > 1 else abs(mu) * 0.15
        return mu, mu - 1.5 * sig, mu + 1.5 * sig

    x     = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted    = slope * x + intercept
    residuals = values - fitted
    rmse      = float(np.sqrt(np.mean(residuals ** 2)))

    x_new = float(n - 1 + horizon)
    x_bar = float(np.mean(x))
    sx    = float(np.sum((x - x_bar) ** 2))
    se    = rmse * np.sqrt(1.0 + 1.0 / n + (x_new - x_bar) ** 2 / (sx + 1e-9))

    forecast = slope * x_new + intercept
    return float(forecast), float(forecast - T_80 * se), float(forecast + T_80 * se)


def generate_machine_forecasts(monthly_df: pd.DataFrame,
                                horizon: int = 1) -> list[MachineForecast]:
    """
    Forecasts all MTBF/MTTR/health metrics for every machine
    in monthly_df for the given horizon (months ahead).
    """
    results = []
    for mid, grp in monthly_df.groupby("machine_id"):
        grp = grp.sort_values("year_month")
        n   = len(grp)

        scores   = grp["health_score"].values.astype(float)
        mtbf_v   = grp["mtbf_hrs"].values.astype(float)
        mttr_v   = grp["mttr_hrs"].values.astype(float)
        avail_v  = grp["availability_pct"].values.astype(float)
        fails_v  = grp["failures_count"].values.astype(float)
        cost_v   = grp["monthly_downtime_cost"].values.astype(float)

        hs, hs_lo, hs_hi = _forecast_series(scores, horizon)
        mt, *_           = _forecast_series(mtbf_v, horizon)
        mr, *_           = _forecast_series(mttr_v, horizon)
        av, *_           = _forecast_series(avail_v, horizon)
        fa, *_           = _forecast_series(fails_v, horizon)
        co, *_           = _forecast_series(cost_v, horizon)

        slope = float(np.polyfit(np.arange(n), scores, 1)[0]) if n >= 3 else 0.0

        # ── Model fit metrics on historical health score series ───────────────
        r2_adj, mae = 0.0, 0.0
        if n >= 3:
            x       = np.arange(n, dtype=float)
            s, ic   = np.polyfit(x, scores, 1)
            fitted  = s * x + ic
            ss_res  = float(np.sum((scores - fitted) ** 2))
            ss_tot  = float(np.sum((scores - scores.mean()) ** 2))
            r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
            r2_adj  = round(1.0 - (1.0 - r2) * (n - 1) / max(n - 2, 1), 3)
            mae     = round(float(np.mean(np.abs(scores - fitted))), 2)

        results.append(MachineForecast(
            machine_id    = str(mid),
            vsm           = grp["vsm"].iloc[0],
            area          = grp["area"].iloc[0],
            horizon       = horizon,
            health_score  = round(clamp(hs), 1),
            health_lo     = round(clamp(hs_lo), 1),
            health_hi     = round(min(100, hs_hi), 1),
            mtbf_hrs      = round(max(0, mt), 1),
            mttr_hrs      = round(max(0, mr), 2),
            availability  = round(clamp(av, 0, 100), 1),
            failures      = round(max(0, fa), 1),
            downtime_cost = round(max(0, co), 0),
            health_slope  = round(slope, 2),
            n_months      = n,
            reliable      = n >= 6,
            r2_adj        = r2_adj,
            mae           = mae,
        ))

    log.info(f"Forecasts generated: {len(results)} machines, horizon={horizon}mo")
    return results


def forecasts_to_df(forecasts: list[MachineForecast]) -> pd.DataFrame:
    return pd.DataFrame([vars(f) for f in forecasts])


def vsm_forecast_summary(forecasts: list[MachineForecast]) -> pd.DataFrame:
    """Aggregate machine forecasts to VSM level."""
    df = forecasts_to_df(forecasts)
    return (
        df.groupby("vsm")
        .agg(
            avg_health_score =("health_score",  "mean"),
            avg_mtbf         =("mtbf_hrs",       "mean"),
            avg_mttr         =("mttr_hrs",       "mean"),
            avg_availability =("availability",   "mean"),
            total_failures   =("failures",       "sum"),
            total_cost       =("downtime_cost",  "sum"),
            avg_slope        =("health_slope",   "mean"),
        )
        .round({"avg_health_score": 1, "avg_mtbf": 1, "avg_mttr": 2,
                "avg_availability": 1, "total_failures": 0,
                "total_cost": 0, "avg_slope": 2})
        .reset_index()
    )
