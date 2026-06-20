# README.md

# AI-Driven Parking Intelligence System

## Objective

Build an end-to-end AI-Driven Parking Intelligence System that transforms historical parking violation records into actionable enforcement intelligence.

The system must identify illegal parking hotspots, analyze their temporal persistence, quantify their operational severity, and dynamically prioritize enforcement zones.

The final system should enable authorities to transition from reactive patrol-based enforcement to proactive, data-driven deployment.

This project intentionally DOES NOT attempt true traffic congestion prediction because the dataset does not contain traffic-flow ground truth such as road speeds, queue lengths, or lane occupancy measurements.

Instead, the system will estimate parking-induced road obstruction risk and convert it into actionable enforcement priorities.

---

# Architectural Decision

Build a:

**Spatio-Temporal Enforcement Priority Intelligence System**

Core components:

1. Data Cleaning Pipeline
2. Spatial Intelligence Engine
3. Temporal Intelligence Engine
4. Enforcement Priority Engine
5. Hotspot Discovery Model (HDBSCAN)
6. Visualization & Dashboard Deliverables

Optional:

7. Predictive Forecasting Layer (only if sufficient historical depth exists)

Do not deviate from this architecture.

Do not propose alternative project directions.

---

# Project Pipeline

Implement every stage sequentially.

---

# Stage 1: Raw Dataset Inspection

## Objective

Understand the structure of the raw dataset before any transformation.

For every feature:

Determine:

* column name
* inferred meaning
* datatype
* percentage missing
* unique value count
* whether it is useful
* whether it should be removed

Generate a dataset summary report.

---

# Stage 2: Data Cleaning

## 2.1 Remove Leakage Features

Remove any columns representing post-event administrative actions.

Examples:

* updated_vehicle_number
* updated_vehicle_type
* validation_status
* validation_timestamp
* data_sent_to_scita_timestamp

Reason:

These variables occur after the parking violation event and introduce future information leakage.

---

## 2.2 Remove Empty Columns

Drop columns with 100% missing values.

Examples:

* description
* closed_datetime
* action_taken_timestamp

Reason:

They contribute no information.

---

## 2.3 Duplicate Handling

Do not blindly remove duplicates.

Duplicate rows should only be removed if all event-defining columns match.

Example:

timestamp
vehicle number
latitude
longitude
violation type

Reason:

Multiple violations occurring at the same location are legitimate observations.

Generate a duplicate audit report.

---

## 2.4 Missing Value Handling

Numerical columns:

Do not use arbitrary mean imputation.

Apply domain-aware strategies.

Categorical columns:

Fill low-missing columns with:

Unknown

Temporal columns:

Never impute timestamps.

Rows missing critical timestamps should be dropped.

Coordinate columns:

Rows missing latitude or longitude should be removed.

Reason:

Spatial intelligence cannot function without coordinates.

---

## 2.5 Data Validation

Check:

### Coordinate validity

latitude ∈ [-90,90]

longitude ∈ [-180,180]

### Temporal validity

modified_datetime >= created_datetime

Invalid rows should be flagged.

Do not automatically remove them unless impossible.

Generate a data quality report.

---

# Stage 3: Temporal Feature Engineering

Convert created_datetime into:

hour

day_of_week

month

is_weekend

is_peak_hour

is_night

Define:

Morning Peak = 7-10

Evening Peak = 17-20

Night = 22-5

Reason:

Parking patterns are highly time-dependent.

---

# Stage 4: Spatial Feature Engineering

Do not use raw latitude and longitude directly.

Convert coordinates into stable geographical zones.

Use:

H3 indexing

Resolution: 8 or empirically optimized

Generate:

h3_cell_id

Reason:

Coordinates are too granular and noisy.

H3 creates consistent geographical boundaries.

---

# Stage 5: Violation Feature Engineering

Process violation_type.

If multiple violations exist:

Convert them into multi-label format.

Apply multi-hot encoding.

Generate:

is_wrong_parking

is_double_parking

is_bus_stop_violation

etc.

Reason:

One event may contain multiple infractions.

---

# Stage 6: Zone-Time Aggregation

Aggregate data at:

H3 cell + hour

Generate:

violation_count

unique_vehicle_count

violation_diversity

vehicle_type_diversity

repeat_offender_ratio

night_violation_ratio

peak_hour_ratio

days_active

weekly_persistence

Reason:

Enforcement decisions operate on zones and time periods, not individual rows.

This becomes the primary analytical table.

---

# Stage 7: Spatial Hotspot Discovery

Model:

HDBSCAN

Inputs:

latitude

longitude

or H3 centroids

Reason:

Parking hotspots are irregular and variable-density.

Do NOT use K-Means.

Generate:

cluster_id

noise_label

cluster_size

cluster_persistence

Outputs:

hotspot boundaries

hotspot centroids

---

# Stage 8: Temporal Pattern Discovery

For every hotspot compute:

peak violation hour

peak violation day

average daily violations

hourly distribution

weekly persistence

Reason:

A hotspot only becomes actionable when authorities know when it recurs.

---

# Stage 9: Enforcement Priority Engine

This is the core intelligence layer.

Build a deterministic scoring system.

DO NOT train a model here.

Construct:

Enforcement Priority Score

Range:

0-100

Formula:

Priority Score =

0.40 × Violation Density

*

0.30 × Temporal Persistence

*

0.20 × Severity Score

*

0.10 × Repeat Offender Ratio

Normalize all components before aggregation.

---

## Component Definitions

### Violation Density

Number of violations per H3 cell.

### Temporal Persistence

Frequency of hotspot occurrence over rolling windows.

### Severity Score

Weight violations.

Example:

Wrong Parking = 1

Double Parking = 2

Bus Stop Obstruction = 3

Adjust according to available data.

### Repeat Offender Ratio

Repeated vehicle appearances.

If unavailable:

Set to zero.

Do not fabricate data.

---

# Stage 10: Optional Forecasting Layer

Only implement if sufficient historical depth exists.

Minimum requirement:

Several weeks or months of timestamp coverage.

Goal:

Forecast future enforcement priority scores.

Model:

CatBoost Regressor

Inputs:

Historical zone aggregates.

Outputs:

Future priority score.

If insufficient data exists:

Skip this stage entirely.

---

# Stage 11: Explainability Layer

Generate feature importance.

Use:

SHAP

Explain:

Why zones became high priority.

Authorities must understand recommendations.

Black-box outputs are unacceptable.

---

# Stage 12: Final Deliverables

The system must generate:

## Deliverable 1

Dynamic hotspot heatmap.

---

## Deliverable 2

Top-N enforcement priority zones.

Columns:

zone

junction

police station

priority score

peak hour

risk level

---

## Deliverable 3

Temporal analytics dashboard.

Visualizations:

Hourly violations

Weekly patterns

Vehicle distribution

Violation distribution

---

## Deliverable 4

Hotspot intelligence report.

For every hotspot:

Location

Severity

Peak hours

Recommended patrol windows

---

# Required Folder Structure

data/

reports/

cleaned/

features/

models/

outputs/

dashboard/

---

# Critical Rules

1. Follow this architecture exactly.
2. Do not invent unavailable data.
3. Do not build true congestion prediction.
4. Every engineering decision must be justified.
5. Prioritize interpretability over complexity.
6. Build an operational intelligence system, not an academic experiment.
7. Every output must help authorities answer one question:

"Where should enforcement officers be deployed, and when?"
