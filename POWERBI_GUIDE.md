# 📊 Power BI Integration Guide

Step-by-step guide to building a Power BI dashboard using the exported data from the Smart City Traffic Analytics project.

---

## 📁 Data Files Location

All Power BI-optimized files are in: `outputs/powerbi/`

| File                         | Type      | Purpose                              |
| ---------------------------- | --------- | ------------------------------------ |
| `fact_accidents_time.csv`    | Fact      | Accidents by state + time slot       |
| `fact_accidents_monthly.csv` | Fact      | Accidents by state + month           |
| `dim_states.csv`             | Dimension | State metadata, regions, populations |
| `agg_state_summary.csv`      | Aggregate | Pre-calculated state summaries       |
| `agg_time_summary.csv`       | Aggregate | Pre-calculated time summaries        |
| `agg_month_summary.csv`      | Aggregate | Pre-calculated month summaries       |
| `summary_kpi.csv`            | KPI       | Key performance indicators           |

---

## 🚀 Quick Start

### Step 1: Import Data

1. Open Power BI Desktop
2. Click **Get Data** → **Text/CSV**
3. Navigate to `outputs/powerbi/`
4. Import these files:
   - `fact_accidents_time.csv`
   - `fact_accidents_monthly.csv`
   - `dim_states.csv`
   - `summary_kpi.csv`

### Step 2: Create Relationships

Go to **Model View** and create these relationships:

```
fact_accidents_time.State → dim_states.State (Many-to-One)
fact_accidents_monthly.State → dim_states.State (Many-to-One)
```

### Step 3: Create Visualizations

---

## 📈 Recommended Dashboard Layout

### Page 1: Executive Overview

| Visual              | Type       | Fields                         |
| ------------------- | ---------- | ------------------------------ |
| **Total Accidents** | Card       | `summary_kpi[TotalAccidents]`  |
| **Total Deaths**    | Card       | `summary_kpi[TotalDeaths]`     |
| **Fatality Rate**   | Card       | `summary_kpi[AvgFatalityRate]` |
| **State Map**       | Filled Map | Location: State, Values: Cases |
| **Top 10 States**   | Bar Chart  | State, Cases                   |

### Page 2: Time Analysis

| Visual                | Type      | Fields                                            |
| --------------------- | --------- | ------------------------------------------------- |
| **Accidents by Hour** | Bar Chart | TimeSlot, Sum of Accidents                        |
| **Heatmap**           | Matrix    | Rows: State, Columns: TimeSlot, Values: Accidents |
| **Day vs Night**      | Pie Chart | Category (Day/Night), Accidents                   |

### Page 3: Monthly Trends

| Visual               | Type           | Fields                    |
| -------------------- | -------------- | ------------------------- |
| **Monthly Trend**    | Line Chart     | Month, Sum of Accidents   |
| **State Comparison** | Clustered Bar  | State, Month, Accidents   |
| **Quarter Analysis** | Stacked Column | Quarter, State, Accidents |

---

## 📐 Data Model

```
┌─────────────────────┐     ┌─────────────────────┐
│  dim_states         │     │  summary_kpi        │
├─────────────────────┤     ├─────────────────────┤
│ State (PK)          │     │ TotalAccidents      │
│ Region              │     │ TotalDeaths         │
│ Population          │     │ TotalInjured        │
│ Area                │     │ AvgFatalityRate     │
└─────────┬───────────┘     └─────────────────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  fact_accidents_    │     │  fact_accidents_    │
│  time               │     │  monthly            │
├─────────────────────┤     ├─────────────────────┤
│ State (FK)          │     │ State (FK)          │
│ TimeSlot            │     │ Month               │
│ Accidents           │     │ Year                │
└─────────────────────┘     │ Accidents           │
                            └─────────────────────┘
```

---

## 🧮 Suggested DAX Measures

### Fatality Rate

```dax
Fatality Rate =
DIVIDE(
    SUM('agg_state_summary'[Died]),
    SUM('agg_state_summary'[Cases]),
    0
) * 100
```

### Year-over-Year Change

```dax
YoY Change =
VAR CurrentYear = SUM('fact_accidents_monthly'[Accidents])
VAR PreviousYear = CALCULATE(
    SUM('fact_accidents_monthly'[Accidents]),
    SAMEPERIODLASTYEAR('Calendar'[Date])
)
RETURN DIVIDE(CurrentYear - PreviousYear, PreviousYear, 0) * 100
```

### Peak Hour Indicator

```dax
Peak Hour =
VAR MaxAccidents = MAXX(ALL('fact_accidents_time'), [TotalAccidents])
RETURN IF([TotalAccidents] = MaxAccidents, "⚠️ Peak", "")
```

---

## 🎨 Design Tips

1. **Color Scheme**: Use red for high severity, amber for medium, green for low
2. **Filters**: Add slicers for State, Month, and TimeSlot
3. **Tooltips**: Show additional context on hover
4. **Drill-through**: Enable drill-through from State to detailed analysis

---

## 🔄 Refreshing Data

To update with new analysis:

1. Re-run the export script:

   ```python
   from src.visualization.powerbi_export import PowerBIExporter
   exporter = PowerBIExporter()
   exporter.export_all("outputs/powerbi")
   ```

2. In Power BI, click **Refresh** to reload the CSV files

---

## 📚 Additional Resources

- [Power BI Documentation](https://docs.microsoft.com/power-bi/)
- [DAX Reference](https://docs.microsoft.com/dax/)
