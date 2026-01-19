"""
Analysis Orchestrator Service
Coordinates ML analysis pipeline using existing models
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path to import existing modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.validator import DataValidator


class AnalysisOrchestrator:
    """Orchestrate ML analysis pipeline with existing models"""
    
    ANALYSIS_TYPES = [
        "patterns",      # Peak hours, seasonal trends
        "forecasting",   # Time-series prediction
        "anomaly",       # Outlier detection
        "classification" # Congestion classification
    ]
    
    def __init__(self):
        self.validator = DataValidator()
    
    def run_analysis(
        self,
        filepath: str,
        schema: Dict[str, Any],
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run ML analysis pipeline on the uploaded dataset
        
        Args:
            filepath: Path to the uploaded file
            schema: Detected schema from validator
            analysis_types: List of analyses to run (None = all)
        
        Returns:
            Dictionary with analysis results
        """
        # Load data
        df = self.validator.load_dataset(filepath)
        
        # Get column mapping
        col_map = schema.get("column_mapping", {})
        
        # Determine which analyses to run
        types_to_run = analysis_types or self.ANALYSIS_TYPES
        
        results = {
            "dataset_info": {
                "rows": len(df),
                "columns": list(df.columns),
                "detected_type": schema.get("detected_type", "generic")
            },
            "analyses": {},
            "summary": {},
            "recommendations": []
        }
        
        # Run each analysis type
        for analysis_type in types_to_run:
            if analysis_type == "patterns":
                results["analyses"]["patterns"] = self._analyze_patterns(df, col_map)
            elif analysis_type == "forecasting":
                results["analyses"]["forecasting"] = self._analyze_forecasting(df, col_map)
            elif analysis_type == "anomaly":
                results["analyses"]["anomaly"] = self._analyze_anomalies(df, col_map)
            elif analysis_type == "classification":
                results["analyses"]["classification"] = self._analyze_classification(df, col_map)
        
        # Generate summary
        results["summary"] = self._generate_summary(results["analyses"])
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results["analyses"])
        
        return results
    
    def _analyze_patterns(self, df: pd.DataFrame, col_map: Dict) -> Dict[str, Any]:
        """Analyze patterns in the data"""
        result = {
            "status": "completed",
            "findings": []
        }
        
        try:
            # Time-based patterns
            if col_map.get("date"):
                date_col = col_map["date"]
                count_col = col_map.get("count")
                
                # Try to parse dates
                try:
                    df['_parsed_date'] = pd.to_datetime(df[date_col])
                    
                    # Extract time components
                    if count_col and count_col in df.columns:
                        # Monthly patterns
                        monthly = df.groupby(df['_parsed_date'].dt.month)[count_col].sum()
                        result["monthly_distribution"] = monthly.to_dict()
                        
                        # Find peak month
                        peak_month = monthly.idxmax()
                        result["peak_month"] = int(peak_month)
                        result["findings"].append(f"Peak activity in month {peak_month}")
                        
                        # Day of week patterns
                        if df['_parsed_date'].dt.day.nunique() > 7:
                            daily = df.groupby(df['_parsed_date'].dt.dayofweek)[count_col].sum()
                            result["daily_distribution"] = daily.to_dict()
                            peak_day = daily.idxmax()
                            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            result["peak_day"] = day_names[int(peak_day)]
                            result["findings"].append(f"Peak activity on {day_names[int(peak_day)]}")
                    
                    df.drop('_parsed_date', axis=1, inplace=True)
                except:
                    result["findings"].append("Could not parse date column for time patterns")
            
            # Location-based patterns
            if col_map.get("location"):
                loc_col = col_map["location"]
                count_col = col_map.get("count")
                
                if count_col and count_col in df.columns:
                    location_stats = df.groupby(loc_col)[count_col].agg(['sum', 'mean', 'count'])
                    top_locations = location_stats.nlargest(10, 'sum')
                    result["top_locations"] = top_locations['sum'].to_dict()
                    result["findings"].append(f"Top location by activity: {top_locations.index[0]}")
            
            # Numeric distribution analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats = df[numeric_cols].describe().to_dict()
                result["numeric_statistics"] = stats
                
        except Exception as e:
            result["status"] = "partial"
            result["error"] = str(e)
        
        return result
    
    def _analyze_forecasting(self, df: pd.DataFrame, col_map: Dict) -> Dict[str, Any]:
        """Generate forecasts based on historical data"""
        result = {
            "status": "completed",
            "method": "trend_analysis",
            "forecast": []
        }
        
        try:
            date_col = col_map.get("date")
            count_col = col_map.get("count")
            
            if not (date_col and count_col):
                result["status"] = "skipped"
                result["message"] = "Requires date and count columns for forecasting"
                return result
            
            # Parse dates and aggregate
            df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
            df_valid = df.dropna(subset=['_parsed_date'])
            
            if len(df_valid) < 3:
                result["status"] = "skipped"
                result["message"] = "Insufficient data points for forecasting"
                return result
            
            # Create time series
            ts = df_valid.groupby(df_valid['_parsed_date'].dt.to_period('M'))[count_col].sum()
            
            if len(ts) >= 3:
                # Simple trend analysis
                x = np.arange(len(ts))
                y = ts.values.astype(float)
                
                # Linear regression
                coeffs = np.polyfit(x, y, 1)
                trend = coeffs[0]
                
                result["trend"] = "increasing" if trend > 0 else "decreasing"
                result["trend_slope"] = float(trend)
                result["avg_value"] = float(np.mean(y))
                
                # Simple forecast for next 3 periods
                future_x = np.array([len(ts), len(ts)+1, len(ts)+2])
                forecast_values = np.polyval(coeffs, future_x)
                
                result["forecast"] = [
                    {"period": i+1, "value": max(0, float(v))}
                    for i, v in enumerate(forecast_values)
                ]
                
                result["findings"] = [
                    f"Overall trend is {result['trend']}",
                    f"Average value: {result['avg_value']:.2f}",
                    f"Forecasted next period: {result['forecast'][0]['value']:.2f}"
                ]
            
            df.drop('_parsed_date', axis=1, inplace=True, errors='ignore')
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _analyze_anomalies(self, df: pd.DataFrame, col_map: Dict) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        result = {
            "status": "completed",
            "anomalies": [],
            "outlier_counts": {}
        }
        
        try:
            # Analyze numeric columns for outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                values = df[col].dropna()
                if len(values) < 10:
                    continue
                
                # Z-score method
                mean = values.mean()
                std = values.std()
                if std == 0:
                    continue
                
                z_scores = np.abs((values - mean) / std)
                outlier_mask = z_scores > 3
                outlier_count = outlier_mask.sum()
                
                result["outlier_counts"][col] = int(outlier_count)
                
                if outlier_count > 0:
                    outlier_values = values[outlier_mask].tolist()[:5]
                    result["anomalies"].append({
                        "column": col,
                        "count": int(outlier_count),
                        "sample_values": outlier_values,
                        "threshold": float(mean + 3*std)
                    })
            
            # Location-based anomalies
            loc_col = col_map.get("location")
            count_col = col_map.get("count")
            
            if loc_col and count_col and count_col in df.columns:
                location_agg = df.groupby(loc_col)[count_col].sum()
                mean = location_agg.mean()
                std = location_agg.std()
                
                if std > 0:
                    high_anomalies = location_agg[location_agg > mean + 2*std]
                    if len(high_anomalies) > 0:
                        result["high_activity_locations"] = high_anomalies.to_dict()
                        result["anomalies"].append({
                            "type": "high_activity_regions",
                            "locations": list(high_anomalies.index),
                            "threshold": float(mean + 2*std)
                        })
            
            result["findings"] = [
                f"Found {len(result['anomalies'])} anomaly categories",
                f"Total outliers detected: {sum(result['outlier_counts'].values())}"
            ]
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _analyze_classification(self, df: pd.DataFrame, col_map: Dict) -> Dict[str, Any]:
        """Classify data into categories"""
        result = {
            "status": "completed",
            "classifications": {}
        }
        
        try:
            count_col = col_map.get("count")
            loc_col = col_map.get("location")
            
            if not count_col or count_col not in df.columns:
                result["status"] = "skipped"
                result["message"] = "Requires count column for classification"
                return result
            
            # Congestion/Activity level classification
            values = df[count_col].dropna()
            
            if len(values) > 0:
                q25, q50, q75 = values.quantile([0.25, 0.5, 0.75])
                
                # Define thresholds
                thresholds = {
                    "LOW": (values.min(), q25),
                    "MODERATE": (q25, q50),
                    "HIGH": (q50, q75),
                    "SEVERE": (q75, values.max())
                }
                result["thresholds"] = {k: [float(v[0]), float(v[1])] for k, v in thresholds.items()}
                
                # Classify each value
                def classify(val):
                    if val <= q25: return "LOW"
                    elif val <= q50: return "MODERATE"
                    elif val <= q75: return "HIGH"
                    else: return "SEVERE"
                
                classifications = df[count_col].apply(classify)
                result["distribution"] = classifications.value_counts().to_dict()
                
                # If we have locations, classify by location
                if loc_col and loc_col in df.columns:
                    loc_agg = df.groupby(loc_col)[count_col].sum()
                    loc_classifications = loc_agg.apply(lambda x: classify(x / len(df) * len(loc_agg)))
                    result["location_classifications"] = loc_classifications.to_dict()
                
                # Format distribution nicely
                dist = result['distribution']
                dist_text = ", ".join([f"{k}: {v}" for k, v in dist.items()])
                
                result["findings"] = [
                    f"Classified into 4 activity levels based on quartiles",
                    f"Activity distribution: {dist_text}"
                ]
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _generate_summary(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all analyses"""
        summary = {
            "total_analyses": len(analyses),
            "completed": sum(1 for a in analyses.values() if a.get("status") == "completed"),
            "key_findings": []
        }
        
        # Collect key findings from each analysis
        for analysis_type, result in analyses.items():
            if result.get("findings"):
                summary["key_findings"].extend(result["findings"][:2])  # Top 2 from each
        
        return summary
    
    def _generate_recommendations(self, analyses: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analyses"""
        recommendations = []
        
        # From patterns
        patterns = analyses.get("patterns", {})
        if patterns.get("peak_month"):
            recommendations.append(f"Focus resources during peak month (Month {patterns['peak_month']})")
        if patterns.get("top_locations"):
            top_loc = list(patterns["top_locations"].keys())[0]
            recommendations.append(f"Priority attention needed in {top_loc} (highest activity)")
        
        # From forecasting
        forecasting = analyses.get("forecasting", {})
        if forecasting.get("trend") == "increasing":
            recommendations.append("Activity is trending upward - consider scaling resources")
        elif forecasting.get("trend") == "decreasing":
            recommendations.append("Activity is declining - investigate potential causes")
        
        # From anomalies
        anomalies = analyses.get("anomaly", {})
        if anomalies.get("high_activity_locations"):
            recommendations.append("Investigate high-activity anomaly regions for root causes")
        
        # From classification
        classification = analyses.get("classification", {})
        if classification.get("distribution", {}).get("SEVERE", 0) > 0:
            severe_pct = classification["distribution"]["SEVERE"]
            recommendations.append(f"Address SEVERE level incidents ({severe_pct} identified)")
        
        return recommendations or ["Analysis complete. No critical recommendations at this time."]
