# Value Stream Health Intelligence

**Spatial MTBF/MTTR analytics dashboard for manufacturing operations — bridging Operational Excellence and Predictive Machine Learning.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple.svg)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Business Problem

Manufacturing plants generate thousands of failure events per year, but most reliability and industrial engineers analyze them in flat spreadsheets, losing spatial and temporal context entirely. When a critical machine goes down on the Beta line, you want to *see* it on the plant floor, understand *why* it failed, *predict* when the next failure will happen, and get a *concrete action plan* — not find a row buried in a CSV.

This project answers four questions in one dashboard:

1. **Where?** — Which machines need attention today and where are they on the floor?
2. **Why?** — Which components and failure types are driving the most downtime cost?
3. **When?** — What will the health score look like in 1–3 months?
4. **What to do?** — What specific action should be taken, by whom, at what cost?

---

## Dashboard — 4 Tabs

### Tab 1 — VS Overview
- Interactive Plotly plant heatmap — hover over any machine to see its full health score breakdown (MTBF / Availability / Failures contributions)
- Value Stream Cost Intelligence cards per VSM line (Alpha, Beta, Gamma) with current + forecast scores
- Top Critical and Top Healthiest machine tables

### Tab 2 — Monthly Trends
- Date range selector (From → To) across the full historical window
- VSM trend lines per metric (Health Score, MTBF, MTTR, Availability, Cost)
- Stacked failure bar chart by area with value labels
- Health score calendar heatmap (machine × month)
- Machine snapshot table for any selected month

### Tab 3 — Predictive Simulation
- Linear trend forecast per machine with 80% prediction intervals
- VSM-level forecast summary cards
- Machine Risk Ranking table (HIGH / MEDIUM / LOW)
- Machine Deep Dive: forecast chart + insight card with trend signal quality
- Technical methodology expander explaining model choice and evolution roadmap

### Tab 4 — Root Cause & Prescriptive
- **A** — Component Cost Pareto (80/20 rule)
- **B** — Component × Machine Downtime Heatmap
- **C** — Avg MTTR by Failure Type & Area
- **D** — Prioritized Prescriptive Action Plan (automated, cost-justified)
- **E** — Cost to Act vs Cost if Ignored (3-month projection)
- **F** — Component ROI Summary (Value Stream-wide)

---

## Methodology

### Health Score (0–100)

Each machine receives a composite health score:

```
normalized_mtbf     = mtbf / max_mtbf_in_VS
normalized_avail    = availability_pct / 100
normalized_failures = 1 - (failures_30d / max_failures_in_VS)

health_score = (normalized_mtbf     × 0.50)
             + (normalized_avail    × 0.30)
             + (normalized_failures × 0.20)
             × 100
```

| Score Range | Status   | Color  |
|-------------|----------|--------|
| 70 – 100    | Healthy  | Green  |
| 40 – 69     | Monitor  | Amber  |
| 0 – 39      | Critical | Red    |

### Predictive Model

Linear regression fitted independently on each machine's monthly health score time series. In a well-run maintenance operation, health scores **do not follow a clean linear trend** — every corrective action resets the trajectory. The model measures the *velocity of deterioration between interventions*, not a fixed long-term destiny. See the in-app Technical Methodology expander (Tab 3) for the full justification and model evolution roadmap.

### Prescriptive Engine

Fleet-relative percentile thresholds (p90 / p75 failure frequency and cost) trigger prioritized actions. Each recommendation includes cost to act, projected cost of inaction over 3 months, and ROI. Actions self-calibrate to any dataset size.

---

## KPIs Tracked

| Metric | Description |
|--------|-------------|
| **MTBF** | Mean time between failures per machine |
| **MTTR** | Mean repair time per machine |
| **Availability** | (Available hrs − Downtime) / Available hrs |
| **Failures (30d)** | Count of failures in the rolling 30-day window |
| **Monthly Downtime Cost** | Downtime hrs × cost per hr |
| **Health Score** | Composite 0–100 reliability score |
| **Component MTTR** | Repair time per component type |
| **Part Lead Time** | Days to receive replacement part |
| **ROI of preventive action** | (Cost if ignored − Cost to act) / Cost to act |

---

## Plant Structure

Three independent VSM production lines flowing bottom → top:

```
┌──────────────────────────────────────────────────────────┐
│               FINISHED GOODS WAREHOUSE                   │
├───────────────┬───────────────┬──────────────────────────┤
│  Assembly     │  Assembly     │  Assembly                │
│  Alpha (8)    │  Beta (5)     │  Gamma (3)               │
├───────────────┼───────────────┼──────────────────────────┤
│  Painting     │  Painting     │  Painting                │
│  2PB + CF     │  1PB + CF     │  1PB + CF                │
├───────────────┼───────────────┼──────────────────────────┤
│  Machining    │  Machining    │  Machining               │
│  5 CNC        │  3CNC+Lathe   │  2CNC + VMC              │
├───────────────┴───────────────┴──────────────────────────┤
│                RAW MATERIALS WAREHOUSE                   │
└──────────────────────────────────────────────────────────┘
    VSM Alpha        VSM Beta         VSM Gamma
  (High-Speed)    (Semi-Auto)        (Flexible)
```

35 machines total across 3 areas (Machining, Painting, Assembly).

---

## Data Sources

Place raw files in `data/raw/` before running the ETL pipeline.

### `equipment_master.csv`
Exported from your ERP or CMMS (SAP PM, Maximo, eMaint).

| Column | Type | Description |
|--------|------|-------------|
| `machine_id` | string | Unique equipment identifier |
| `vsm` | string | Production line (Alpha / Beta / Gamma) |
| `area` | string | Plant area (Machining / Painting / Assembly) |
| `machine_type` | string | Equipment category (CNC, Lathe, VMC, Paint Booth, etc.) |
| `year_installed` | integer | Year of installation |
| `manufacturer` | string | Equipment manufacturer |
| `downtime_cost_per_hr` | float | Estimated cost per hour of unplanned downtime (USD) |

### `failures.csv`
Exported from your CMMS work order system (corrective maintenance orders).

| Column | Type | Description |
|--------|------|-------------|
| `failure_id` | string | Unique work order ID |
| `machine_id` | string | Equipment that failed |
| `failure_date` | date | Date the failure occurred (YYYY-MM-DD) |
| `failure_mode` | string | What failed (e.g. "Spindle fault", "Coolant leak") |
| `downtime_hrs` | float | Total hours the machine was out of service |
| `repair_hrs` | float | Hours spent on the repair |
| `technician_id` | string | Technician who performed the repair |
| `root_cause` | string | Root cause classification |
| `corrective_action` | string | Action taken to resolve |
| `component` | string | Subsystem that failed (spindle, servo_drive, coolant_system, etc.) |
| `failure_type` | string | mechanical / electrical / software / hydraulic |
| `technician_type` | string | mechanical / electrical / automation / hydraulic |
| `part_replaced` | string | Part number or description replaced |
| `part_cost_usd` | float | Cost of the replacement part (USD) |
| `part_lead_time_days` | integer | Days to receive the part from supplier |
| `time_to_diagnose_hrs` | float | Hours spent on diagnosis before repair |

### `production_data.csv`
Exported from your MES or shift report system.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Production date (YYYY-MM-DD) |
| `vsm` | string | Production line |
| `shift` | string | Shift (Day / Evening / Night) |
| `planned_hrs` | float | Planned production hours |
| `actual_hrs` | float | Actual hours run |
| `units_produced` | integer | Units completed |
| `units_rejected` | integer | Units rejected / scrapped |
| `oee_pct` | float | Overall Equipment Effectiveness (%) |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your data files

```
data/raw/
├── equipment_master.csv
├── failures.csv
└── production_data.csv
```

### 3. Run the ETL pipeline

```bash
python _run_etl.py
```

Generates three processed files:

```
data/processed/
├── mtbf_metrics.csv       # Current-period KPIs per machine
├── monthly_metrics.csv    # Time-series (machine × month) for forecasting
└── component_metrics.csv  # Component-level failure aggregations for root cause
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Project Structure

```
mtbf_spatial_analytics/
├── app.py                    # Streamlit dashboard — 4 tabs
├── config.yaml               # Path and parameter configuration
├── requirements.txt
├── _run_etl.py               # ETL runner (generates all processed CSVs)
│
├── data/
│   ├── raw/                  # Source CSVs (equipment, failures, production)
│   ├── processed/            # ETL output (3 CSVs)
│   └── layout/               # zones.json + plant_layout.png
│
├── src/
│   ├── etl/                  # Extract / Transform / Load pipeline
│   │   ├── extract.py
│   │   ├── transform.py      # MTBF, monthly, and component metrics
│   │   └── load.py
│   ├── metrics/              # MTBF, MTTR, OEE calculation modules
│   ├── spatial/              # Coordinate mapping + Plotly interactive heatmap
│   │   ├── coordinates.py
│   │   ├── heatmap.py        # generate_heatmap_plotly() with hover tooltips
│   │   └── polygons.py
│   ├── visualization/        # Plot helpers per tab
│   │   ├── plots.py          # KPI tables (VSM summary, top critical/healthy)
│   │   ├── trends.py         # Monthly trend charts + machine forecast chart
│   │   └── rootcause.py      # Pareto, heatmap, MTTR, urgency charts
│   ├── ml/
│   │   ├── forecasting.py    # Linear regression per machine + R²adj / MAE
│   │   ├── explainer.py      # Machine insight text + risk table
│   │   └── prescriptive.py   # Prescriptive action engine + ROI calculation
│   └── utils/                # Config, logger, helpers
│
├── notebooks/                # Jupyter EDA + methodology walkthroughs
├── tests/                    # Pytest suite
└── outputs/figures/          # Generated heatmap PNG
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Roadmap

- [x] Phase 1: MTBF/MTTR spatial heatmap dashboard
- [x] Phase 2: Monthly trends, date range analysis, health score calendar
- [x] Phase 3: Predictive simulation — linear trend forecast with 80% CI per machine
- [x] Phase 4: Root cause & prescriptive analytics — component-level intelligence, automated action plan
- [x] Phase 5: Interactive Plotly heatmap with hover score breakdown
- [ ] Phase 6: Holt-Winters / ARIMA upgrade (recommended at 18+ months of data)
- [ ] Phase 7: Real-time sensor data integration (MQTT / OPC-UA)

---

*LozanoLsa · Turning Operations into Predictive Systems*
