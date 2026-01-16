# 📚 Technical Documentation

API reference and detailed documentation for the Smart City Traffic Analytics modules.

---

## Table of Contents

- [Analysis Module](#analysis-module)
- [Models Module](#models-module)
- [Visualization Module](#visualization-module)
- [Data Dictionary](#data-dictionary)

---

## Analysis Module

### `src/analysis/patterns.py`

#### PatternRecognizer

Analyzes traffic patterns from accident data.

```python
from src.analysis.patterns import PatternRecognizer

analyzer = PatternRecognizer()
analyzer.load_processed_data()
```

**Methods:**

| Method                          | Parameters                | Returns   | Description               |
| ------------------------------- | ------------------------- | --------- | ------------------------- |
| `load_processed_data()`         | `time_file`, `month_file` | None      | Load processed CSV files  |
| `analyze_peak_hours()`          | `state: Optional[str]`    | DataFrame | Get peak accident hours   |
| `analyze_monthly_patterns()`    | `state: Optional[str]`    | DataFrame | Monthly pattern analysis  |
| `get_state_comparison()`        | -                         | DataFrame | State-by-state comparison |
| `get_peak_hour_by_state()`      | -                         | DataFrame | Peak hour for each state  |
| `get_day_vs_night_comparison()` | -                         | Dict      | Day vs night statistics   |
| `generate_pattern_summary()`    | -                         | Dict      | Overall summary metrics   |

**Example:**

```python
analyzer = PatternRecognizer()
analyzer.load_processed_data()

# Get peak hours nationally
peak_hours = analyzer.analyze_peak_hours()
print(peak_hours.head())

# Get peak hours for specific state
peak_hours_tn = analyzer.analyze_peak_hours(state="Tamil Nadu")
```

---

### `src/analysis/bottlenecks.py`

#### BottleneckAnalyzer

Identifies traffic bottlenecks and congestion hotspots.

**Methods:**

| Method                        | Parameters  | Returns   | Description                    |
| ----------------------------- | ----------- | --------- | ------------------------------ |
| `load_processed_data()`       | file paths  | None      | Load datasets                  |
| `identify_bottlenecks()`      | `threshold` | DataFrame | Find chronic congestion points |
| `rank_states_by_congestion()` | -           | DataFrame | Rank states by issues          |
| `get_bottleneck_summary()`    | -           | Dict      | Summary statistics             |

---

### `src/analysis/time_series.py`

#### TimeSeriesAnalyzer

Performs time-series decomposition and trend analysis.

**Methods:**

| Method                       | Parameters | Returns   | Description                          |
| ---------------------------- | ---------- | --------- | ------------------------------------ |
| `load_processed_data()`      | file paths | None      | Load time-series data                |
| `decompose_trend()`          | `state`    | Dict      | Trend/seasonality decomposition      |
| `calculate_moving_average()` | `window`   | DataFrame | Rolling averages                     |
| `detect_trend_direction()`   | -          | str       | 'increasing', 'decreasing', 'stable' |

---

## Models Module

### `src/models/forecasting.py`

#### TrafficForecaster

Forecast traffic accidents using ML models.

```python
from src.models.forecasting import TrafficForecaster

forecaster = TrafficForecaster()
forecaster.load_processed_data()
```

**Methods:**

| Method                      | Parameters               | Returns        | Description                |
| --------------------------- | ------------------------ | -------------- | -------------------------- |
| `load_processed_data()`     | file paths               | None           | Load training data         |
| `prepare_features()`        | `state`                  | DataFrame      | Feature engineering        |
| `train_linear_regression()` | `state`                  | Dict (metrics) | Train Linear Regression    |
| `train_random_forest()`     | `state`                  | Dict (metrics) | Train Random Forest        |
| `forecast_next_months()`    | `n_months`, `model_name` | DataFrame      | Predict future accidents   |
| `compare_models()`          | -                        | DataFrame      | Compare model performance  |
| `get_feature_importance()`  | -                        | DataFrame      | Feature importance from RF |

**Example:**

```python
forecaster = TrafficForecaster()
forecaster.load_processed_data()

# Train models
lr_metrics = forecaster.train_linear_regression()
rf_metrics = forecaster.train_random_forest()

# Forecast next 3 months
predictions = forecaster.forecast_next_months(n_months=3)
print(predictions)
```

---

### `src/models/classification.py`

#### SeverityClassifier

Classifies accident severity into categories.

**Methods:**

| Method                        | Parameters       | Returns | Description            |
| ----------------------------- | ---------------- | ------- | ---------------------- |
| `train()`                     | features, labels | Dict    | Train classifier       |
| `predict()`                   | features         | array   | Predict severity class |
| `get_classification_report()` | -                | str     | Detailed metrics       |

---

### `src/models/anomaly.py`

#### AnomalyDetector

Detects unusual patterns in traffic data.

**Methods:**

| Method                | Parameters | Returns   | Description    |
| --------------------- | ---------- | --------- | -------------- |
| `fit()`               | data       | None      | Fit detector   |
| `detect()`            | data       | DataFrame | Find anomalies |
| `get_anomaly_score()` | data       | array     | Anomaly scores |

---

## Visualization Module

### `src/visualization/plots.py`

#### TrafficPlotter

Generate visualizations using Plotly and Matplotlib.

**Methods:**

| Method                       | Parameters      | Returns | Description         |
| ---------------------------- | --------------- | ------- | ------------------- |
| `plot_time_distribution()`   | data, save_path | Figure  | Bar chart by time   |
| `plot_monthly_trend()`       | data, save_path | Figure  | Line chart by month |
| `plot_heatmap()`             | data, save_path | Figure  | State×Time heatmap  |
| `plot_severity_comparison()` | data, save_path | Figure  | Severity bar chart  |

---

### `src/visualization/powerbi_export.py`

#### PowerBIExporter

Export data optimized for Power BI dashboards.

```python
from src.visualization.powerbi_export import PowerBIExporter

exporter = PowerBIExporter()
exporter.export_all(output_dir="outputs/powerbi")
```

**Methods:**

| Method                      | Parameters | Returns | Description                |
| --------------------------- | ---------- | ------- | -------------------------- |
| `export_fact_tables()`      | output_dir | None    | Export fact tables         |
| `export_dimension_tables()` | output_dir | None    | Export dimension tables    |
| `export_aggregations()`     | output_dir | None    | Export pre-aggregated data |
| `export_kpis()`             | output_dir | None    | Export KPI summary         |
| `export_all()`              | output_dir | None    | Export everything          |

---

## Data Dictionary

### Processed Datasets

#### `accidents_time_processed.csv`

| Column    | Type   | Description                              |
| --------- | ------ | ---------------------------------------- |
| State     | string | State/UT name                            |
| TimeSlot  | string | 3-hour time window (e.g., "06:00-09:00") |
| Accidents | int    | Number of accidents                      |

#### `accidents_month_processed.csv`

| Column    | Type   | Description         |
| --------- | ------ | ------------------- |
| State     | string | State/UT name       |
| Month     | string | Month name          |
| Year      | int    | Year (2023)         |
| Accidents | int    | Number of accidents |

#### `accidents_severity_processed.csv`

| Column       | Type   | Description                |
| ------------ | ------ | -------------------------- |
| State        | string | State/UT name              |
| Cases        | int    | Total accident cases       |
| Injured      | int    | Number injured             |
| Died         | int    | Number killed              |
| FatalityRate | float  | Deaths per 100 accidents   |
| InjuryRate   | float  | Injuries per 100 accidents |

#### `vehicle_registrations_processed.csv`

| Column      | Type   | Description         |
| ----------- | ------ | ------------------- |
| State       | string | State/UT name       |
| VehicleType | string | Type of vehicle     |
| Count       | int    | Registered vehicles |
