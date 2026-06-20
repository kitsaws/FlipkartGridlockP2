# Parking Violation Hotspot Intelligence System (V2)

## Overview

This project predicts and explains parking violation hotspots using historical violation records.

The original version of the project aggregated data by:

```text
Location
Hour
Day
Month
```

and trained a hotspot classification model.

While this successfully identified *where* hotspots occur, it could not explain *why* they occur because violation categories were ignored during aggregation.

The upgraded architecture introduces **Violation-Aware Hotspot Prediction**, allowing the system to:

- Predict hotspot risk levels
- Identify the violation types contributing to hotspot formation
- Filter hotspot predictions by violation category
- Provide explainable predictions using SHAP
- Support future continuous learning through periodic retraining

---

# Objectives

## Existing Goals

- Identify parking congestion hotspots
- Predict hotspot risk levels
- Explain model decisions

## New Goals

- Make hotspot predictions violation-aware
- Allow enforcement agencies to filter hotspots by violation category
- Improve explainability
- Prioritize dangerous violations over minor ones
- Support future live-data retraining

---

# High-Level Architecture

```text
Raw Violation Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Violation-Aware Aggregation
        ↓
Weighted Hotspot Score Generation
        ↓
Risk Classification Model
        ↓
SHAP Explainability
        ↓
Dashboard & API
        ↓
Continuous Retraining Pipeline
```

---

# Pipeline Design

## Stage 1 — Dataset Inspection

File:

```text
01_dataset_inspection.py
```

Purpose:

- Understand schema
- Identify missing values
- Analyze violation categories
- Analyze location distribution

Outputs:

- Data quality report
- Dataset statistics

---

## Stage 2 — Data Cleaning

File:

```text
02_data_cleaning.py
```

Purpose:

- Remove duplicates
- Handle missing values
- Normalize location fields
- Standardize violation categories

Outputs:

Cleaned dataset.

---

## Stage 3 — Feature Engineering

File:

```text
03_feature_engineering.py
```

Purpose:

Extract temporal features:

```text
Hour
Day
Weekday
Month
```

Create location features.

Outputs:

Feature-enhanced dataset.

---

# NEW ARCHITECTURE

Everything below this stage is redesigned.

---

# Stage 4 — Violation-Aware Aggregation

File:

```text
04_violation_aware_aggregation.py
```

## Previous Aggregation

```python
groupby(
    Location,
    Hour,
    Day,
    Month
)
```

Example:

| Location | Hour | Day | Month | Violations |
|----------|------|------|--------|------------|
| A | 9 | Mon | Jan | 25 |

This loses information regarding violation types.

---

## New Aggregation

```python
groupby(
    Location,
    Hour,
    Day,
    Month,
    Violation_Type
)
```

Example:

Raw Records:

| Location | Time | Violation |
|----------|------|-----------|
| A | 9:01 | Double Parking |
| A | 9:05 | Double Parking |
| A | 9:10 | Footpath Parking |
| A | 9:15 | Footpath Parking |
| A | 9:20 | Zebra Crossing |

Aggregated Output:

| Location | Hour | Day | Month | Violation Type | Count |
|----------|------|------|--------|----------------|--------|
| A | 9 | Mon | Jan | Double Parking | 2 |
| A | 9 | Mon | Jan | Footpath Parking | 2 |
| A | 9 | Mon | Jan | Zebra Crossing | 1 |

Benefits:

- Preserves violation composition
- Enables violation-specific hotspot discovery
- Improves explainability

---

# Stage 5 — Weighted Hotspot Score Generation

File:

```text
05_weighted_hotspot_scoring.py
```

Not all violations are equally dangerous.

Example severity weights:

| Violation Type | Weight |
|---------------|---------|
| Double Parking | 1.0 |
| Footpath Parking | 1.5 |
| Near Traffic Signal | 2.0 |
| Zebra Crossing | 3.0 |

---

## Example

Aggregated Data:

| Violation | Count |
|-----------|--------|
| Double Parking | 20 |
| Zebra Crossing | 5 |

Simple Count:

```text
20 + 5 = 25
```

Weighted Score:

```text
(20 × 1.0) + (5 × 3.0)
= 35
```

This weighted score becomes the hotspot intensity indicator.

---

# Stage 6 — Risk Label Generation

File:

```text
06_risk_label_generation.py
```

Convert hotspot scores into classes.

Example:

| Score | Risk |
|---------|---------|
| 0–10 | Low |
| 11–30 | Medium |
| >30 | High |

Generated target column:

```text
Risk_Class
```

Possible values:

```text
Low
Medium
High
```

---

# Stage 7 — Hotspot Risk Classification Model

File:

```text
11_risk_classification_model.py
```

## Features

```text
Location
Hour
Day
Month
Violation_Type
Weighted_Hotspot_Score
```

Target:

```text
Risk_Class
```

---

## Example

| Location | Hour | Violation | Score | Risk |
|----------|------|-----------|--------|--------|
| A | 9 | Zebra Crossing | 40 | High |
| B | 9 | Double Parking | 10 | Medium |
| C | 11 | Footpath Parking | 2 | Low |

The model learns:

```text
Certain violations at certain locations and times
are historically associated with hotspot formation.
```

---

# Stage 8 — Explainable AI (SHAP)

File:

```text
14_explain_predictions.py
```

Purpose:

Explain WHY a hotspot prediction was made.

Example Output:

```text
Prediction:
HIGH RISK
```

Top contributing factors:

```text
Location A → +40%
Zebra Crossing Violations → +35%
Morning Rush Hour → +20%
```

Benefits:

- Improves transparency
- Helps traffic authorities understand causes
- Improves trust in predictions

---

# Stage 9 — Dashboard Filtering

Frontend Upgrade

New filter:

```text
Violation Type
```

Options:

```text
All Violations
Double Parking
Footpath Parking
Traffic Signal Area
Zebra Crossing
```

---

## Example Use Cases

### Scenario 1

User selects:

```text
All Violations
```

Output:

```text
Top Overall Hotspots
```

---

### Scenario 2

User selects:

```text
Double Parking
```

Output:

```text
Double Parking Hotspots
```

---

### Scenario 3

User selects:

```text
Zebra Crossing
```

Output:

```text
Pedestrian Safety Hotspots
```

This creates actionable enforcement insights.

---

# Stage 10 — Model Evaluation

File:

```text
15_evaluate_models.py
```

Metrics:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
```

Primary objective:

Accurate hotspot classification.

---

# Removed Component

## Severity Regression Model

Previous File:

```text
12_severity_learning.py
```

Status:

```text
Deprecated
```

Reason:

- Poor performance
- Adds unnecessary complexity
- Risk classes provide sufficient operational value

Instead of predicting:

```text
Severity = 27.63
```

We predict:

```text
Low
Medium
High
```

which is easier to explain and more useful operationally.

---

# Continuous Learning Roadmap

Current Version:

```text
Historical Data
        ↓
Training
        ↓
Prediction
```

Future Version:

```text
Historical Data
        +
New Daily Data
        ↓
Aggregation
        ↓
Dataset Update
        ↓
Model Retraining
        ↓
Updated Predictions
```

Retraining Frequency:

```text
Daily
Weekly
Monthly
```

depending on deployment requirements.

---

# Deliverables

## Core Outputs

- Hotspot Risk Prediction
- Violation-Specific Hotspot Discovery
- SHAP Explainability
- Interactive Dashboard

## User Capabilities

- View hotspot maps
- Filter by violation category
- Inspect hotspot causes
- Prioritize enforcement resources

---

# Expected Benefits

### Before Upgrade

```text
Location A is a hotspot.
```

### After Upgrade

```text
Location A is a hotspot primarily due to
Zebra Crossing and Double Parking violations
during morning rush hours.
```

This transforms the system from a simple hotspot detector into an actionable parking enforcement intelligence platform.