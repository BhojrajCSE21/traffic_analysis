# Traffic Analytics Platform

A modern web application that allows users to upload traffic/transportation datasets and receive ML-powered analysis with interactive visualizations.

## Features

- **Dataset Upload**: Drag & drop CSV/Excel files (up to 50MB)
- **Auto Schema Detection**: Automatically identifies date, location, count, and severity columns
- **ML Analysis Pipeline**:
  - Pattern Recognition (peak hours, seasonal trends, top locations)
  - Time-Series Forecasting (trend detection, future predictions)
  - Anomaly Detection (outliers, high-activity regions)
  - Classification (activity level categorization)
- **Interactive Charts**: Plotly-based visualizations
- **Export Results**: Download analysis as JSON

## Quick Start

```bash
# Make run script executable
chmod +x run.sh

# Start the platform
./run.sh
```

Then open http://localhost:8000 in your browser.

## API Endpoints

| Endpoint            | Method | Description                 |
| ------------------- | ------ | --------------------------- |
| `/api/upload`       | POST   | Upload a dataset file       |
| `/api/datasets`     | GET    | List all uploaded datasets  |
| `/api/analyze/{id}` | POST   | Start analysis on a dataset |
| `/api/results/{id}` | GET    | Get analysis results        |
| `/api/charts/{id}`  | GET    | Get generated chart files   |

API Documentation: http://localhost:8000/docs

## Tech Stack

- **Backend**: FastAPI, Python
- **ML**: scikit-learn, pandas, numpy
- **Visualization**: Plotly
- **Frontend**: Vanilla JS, CSS (Dark Theme with Glassmorphism)

## Project Structure

```
platform/
├── backend/
│   ├── main.py              # FastAPI application
│   └── services/
│       ├── validator.py     # Data validation & schema detection
│       ├── orchestrator.py  # ML pipeline coordination
│       └── visualization.py # Chart generation
├── frontend/
│   ├── index.html           # Main page
│   ├── styles/main.css      # Premium dark theme
│   └── js/app.js            # Application logic
├── uploads/                  # Uploaded files (auto-created)
├── results/                  # Analysis results & charts (auto-created)
├── requirements.txt
├── run.sh
└── README.md
```
