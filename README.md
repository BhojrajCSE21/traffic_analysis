# 🚦 Smart City Traffic & Transportation Analytics

A comprehensive traffic analytics system analyzing Indian traffic accident data (2023) to identify patterns, detect bottlenecks, forecast trends, and provide actionable insights for urban transportation planning.

---

## 📈 Executive Summary

### Key Findings from Analysis

| Metric                       | Value   |
| ---------------------------- | ------- |
| **Total Accidents Analyzed** | 563,011 |
| **Total Fatalities**         | 215,782 |
| **Total Injuries**           | 515,210 |
| **Average Fatality Rate**    | 41.79%  |

### Top 5 States by Accident Cases

1. **Tamil Nadu** - 69,491 cases (29.18% fatality rate)
2. **Madhya Pradesh** - 56,475 cases (27.99% fatality rate)
3. **Kerala** - 46,392 cases (9.69% fatality rate)
4. **Karnataka** - 43,440 cases (28.37% fatality rate)
5. **Uttar Pradesh** - 42,001 cases (66.91% fatality rate)

### Peak Accident Hours

- **Highest Risk**: 18:00-21:00 (20.37% of all accidents)
- **Second Highest**: 15:00-18:00 (17.19% of all accidents)
- **Safest Window**: 00:00-06:00 (11.15% combined)

---

## 🎯 Problem Statement

Urban traffic congestion in Indian metropolitan cities causes:

- **₹1.47 lakh crore** annual economic losses
- **30%** increase in vehicle emissions during peak congestion
- Average **2 hours/day** wasted by commuters in traffic

---

## 📁 Project Structure

```
traffic_analysis/
├── config/                     # Configuration settings
│   └── settings.py             # Paths and parameters
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned & transformed data
│   └── external/               # External reference data
├── src/
│   ├── data/                   # Data loading & preprocessing
│   │   ├── loader.py           # Dataset loading utilities
│   │   └── preprocessor.py     # Data cleaning & transformation
│   ├── analysis/               # Core analysis modules
│   │   ├── patterns.py         # Peak hour & seasonal pattern detection
│   │   ├── bottlenecks.py      # Congestion hotspot identification
│   │   └── time_series.py      # Trend decomposition & analysis
│   ├── models/                 # Machine learning models
│   │   ├── forecasting.py      # Traffic prediction (Linear, RF)
│   │   ├── classification.py   # Severity classification
│   │   └── anomaly.py          # Anomaly detection
│   └── visualization/          # Plotting & export
│       ├── plots.py            # Plotly/Matplotlib visualizations
│       └── powerbi_export.py   # Power BI data export
├── dashboard/                  # Dash web application
│   ├── components/             # Reusable UI components
│   └── layouts/                # Page layouts
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_pattern_analysis.ipynb
│   ├── 03_predictive_models.ipynb
│   └── 04_visualization.ipynb
├── outputs/
│   ├── figures/                # Generated visualizations (PNG/HTML)
│   ├── powerbi/                # Power BI optimized exports
│   ├── models/                 # Saved ML models
│   └── reports/                # Analysis reports
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

```bash
# Clone/navigate to project
cd traffic_analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
python dashboard/app.py
# Open browser at http://127.0.0.1:8050
```

### Running Notebooks

```bash
# Start Jupyter
./venv/bin/jupyter notebook notebooks/

# Or view pre-generated HTML outputs
open notebooks/html_output/01_data_exploration.html
```

---

## 🛠️ Technology Stack

| Category            | Technologies                        |
| ------------------- | ----------------------------------- |
| **Data Processing** | Pandas, NumPy                       |
| **Visualization**   | Plotly, Matplotlib, Seaborn, Folium |
| **ML/Forecasting**  | Scikit-learn, Prophet, XGBoost      |
| **Dashboard**       | Dash, Plotly                        |
| **Geospatial**      | GeoPandas, Folium                   |
| **Notebooks**       | Jupyter                             |

---

## 📊 Module Documentation

### Analysis Modules (`src/analysis/`)

| Module           | Key Class            | Purpose                                                      |
| ---------------- | -------------------- | ------------------------------------------------------------ |
| `patterns.py`    | `PatternRecognizer`  | Peak hour detection, seasonal patterns, day/night comparison |
| `bottlenecks.py` | `BottleneckAnalyzer` | Identify chronic congestion hotspots                         |
| `time_series.py` | `TimeSeriesAnalyzer` | Trend decomposition, seasonality analysis                    |

### Prediction Models (`src/models/`)

| Module              | Key Class            | Purpose                                      |
| ------------------- | -------------------- | -------------------------------------------- |
| `forecasting.py`    | `TrafficForecaster`  | Linear Regression, Random Forest predictions |
| `classification.py` | `SeverityClassifier` | Classify accident severity levels            |
| `anomaly.py`        | `AnomalyDetector`    | Detect unusual traffic patterns              |

### Visualization (`src/visualization/`)

| Module              | Key Class         | Purpose                            |
| ------------------- | ----------------- | ---------------------------------- |
| `plots.py`          | `TrafficPlotter`  | Generate Plotly/Matplotlib charts  |
| `powerbi_export.py` | `PowerBIExporter` | Export data optimized for Power BI |

---

## 📓 Jupyter Notebooks

| Notebook                     | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `01_data_exploration.ipynb`  | Data loading, quality assessment, initial statistics |
| `02_pattern_analysis.ipynb`  | Peak hours, seasonal trends, state comparisons       |
| `03_predictive_models.ipynb` | ML model training, forecasting, feature importance   |
| `04_visualization.ipynb`     | Advanced interactive visualizations                  |

**View HTML outputs**: `notebooks/html_output/`

---

## 📤 Output Files

### Figures (`outputs/figures/`)

| File                            | Description                         |
| ------------------------------- | ----------------------------------- |
| `accidents_by_time.png/html`    | Accidents distribution by time slot |
| `accidents_by_month.png/html`   | Monthly trend analysis              |
| `heatmap_state_time.png/html`   | State × Time heatmap                |
| `fatality_vs_cases.png/html`    | Fatality rate vs case volume        |
| `top_states_severity.png/html`  | Top states by severity metrics      |
| `day_night_comparison.png/html` | Day vs night accident patterns      |
| `quarterly_trend.png/html`      | Quarterly trend analysis            |

### Power BI Exports (`outputs/powerbi/`)

| File                         | Description                        |
| ---------------------------- | ---------------------------------- |
| `fact_accidents_time.csv`    | Fact table: accidents by time slot |
| `fact_accidents_monthly.csv` | Fact table: accidents by month     |
| `dim_states.csv`             | Dimension table: state metadata    |
| `agg_state_summary.csv`      | Pre-aggregated state summaries     |
| `agg_time_summary.csv`       | Pre-aggregated time summaries      |
| `agg_month_summary.csv`      | Pre-aggregated month summaries     |
| `summary_kpi.csv/json`       | Key performance indicators         |

---

## 📊 Power BI Integration

See [POWERBI_GUIDE.md](POWERBI_GUIDE.md) for detailed instructions.

**Quick Start:**

1. Import CSV files from `outputs/powerbi/`
2. Create relationships:
   - `fact_accidents_time.State` → `dim_states.State`
   - `fact_accidents_monthly.State` → `dim_states.State`
3. Use `summary_kpi.csv` for KPI cards

---

## 🔬 Methodology

### Data Pipeline

1. **Raw Data** → CSV files from government traffic statistics
2. **Preprocessing** → Cleaning, normalization, feature engineering
3. **Analysis** → Pattern recognition, trend decomposition
4. **Modeling** → Forecasting, classification, anomaly detection
5. **Visualization** → Interactive dashboards, exportable charts

### Machine Learning Approach

- **Forecasting**: Linear Regression & Random Forest for time-series prediction
- **Classification**: Multi-class severity classification
- **Anomaly Detection**: Statistical and ML-based outlier detection

---

## 📜 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.
