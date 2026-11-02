# Daily Demand Time-Series Dataset Provenance & Metadata

## Overview
This directory contains the canonical historical daily demand time-series dataset `daily-demand-series.csv` (and symlinked `demand.csv`), recording daily product demand with pronounced upward trend, weekly seasonality, marketing promotional events, and weekend/holiday factors across June 2026.

## Dataset Specifications
- **File Path**: `data/daily-demand-series.csv`
- **Total Observations ($T$)**: 30 consecutive calendar days (2026-06-01 to 2026-06-30)
- **Temporal Frequency**: Daily (`D`)
- **Variables**:
  1. `date`: Calendar date in ISO 8601 format (`YYYY-MM-DD`).
  2. `demand`: Integer quantity of daily units demanded (range: 118 to 174, mean: 142.10, std: 13.84).
  3. `marketing_event`: Binary flag ($1$ on promotional campaign days—predominantly Fridays and Saturdays; $0$ otherwise).
  4. `holiday`: Binary flag ($1$ on designated holidays / weekend peaks—Sundays; $0$ otherwise).

## Time-Series Structural Characteristics
- **Deterministic Trend**: Strong linear upward trajectory with a daily drift of $\approx +0.97$ units/day (rising from 118 on June 1 to 174 on June 27).
- **Weekly Seasonality ($s = 7$)**: Distinct intra-week cyclic patterns where demand reaches its nadir on Mondays and peaks sharply over Friday-Sunday driven by consumer shopping behavior and scheduled promotions.
- **Holdout Evaluation Period**: Days 24–30 (June 24 to June 30, 2026, $n = 7$) serve as the out-of-sample holdout test partition covering an entire 7-day seasonal cycle.
- **Forecasting Horizon**: 30 days forward into July 2026 (2026-07-01 through 2026-07-30).

## License
Released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
