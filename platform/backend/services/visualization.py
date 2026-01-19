"""
Visualization Service
Generate professional, consistent charts from analysis results
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class VisualizationService:
    """Generate professional visualizations from analysis results"""
    
    # Modern color palette
    COLORS = {
        "primary": "#6366f1",      # Indigo
        "secondary": "#8b5cf6",    # Purple
        "success": "#10b981",      # Emerald
        "warning": "#f59e0b",      # Amber
        "danger": "#ef4444",       # Red
        "info": "#3b82f6",         # Blue
    }
    
    # Order for classification levels
    LEVEL_ORDER = ["LOW", "MODERATE", "HIGH", "SEVERE"]
    LEVEL_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]
    
    # Consistent chart dimensions
    CHART_HEIGHT = 400
    CHART_WIDTH = 550
    
    # Base layout for all charts
    BASE_LAYOUT = {
        "paper_bgcolor": "rgba(10, 10, 15, 1)",
        "plot_bgcolor": "rgba(20, 20, 35, 0.8)",
        "font": {"family": "Inter, sans-serif", "color": "#e2e8f0", "size": 12},
        "margin": {"l": 60, "r": 40, "t": 50, "b": 60},
        "title": {"font": {"size": 16, "color": "#f1f5f9"}},
    }
    
    def __init__(self):
        pass
    
    def generate_charts(
        self,
        results: Dict[str, Any],
        analysis_id: str,
        output_dir: str
    ) -> List[Dict[str, str]]:
        """Generate all charts for the analysis results"""
        charts = []
        
        # Create output directory
        chart_dir = Path(output_dir) / analysis_id
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        analyses = results.get("analyses", {})
        
        # Generate pattern charts
        if "patterns" in analyses:
            charts.extend(self._generate_pattern_charts(analyses["patterns"], chart_dir))
        
        # Generate anomaly charts
        if "anomaly" in analyses:
            charts.extend(self._generate_anomaly_charts(analyses["anomaly"], chart_dir))
        
        # Generate classification charts
        if "classification" in analyses:
            charts.extend(self._generate_classification_charts(analyses["classification"], chart_dir))
        
        # Generate forecasting charts
        if "forecasting" in analyses:
            charts.extend(self._generate_forecast_charts(analyses["forecasting"], chart_dir))
        
        return charts
    
    def _apply_layout(self, fig, title: str, xaxis_title: str = "", yaxis_title: str = ""):
        """Apply consistent layout to a figure"""
        layout = {
            **self.BASE_LAYOUT,
            "title": {"text": title, "font": {"size": 16, "color": "#f1f5f9"}, "x": 0.5, "xanchor": "center"},
            "height": self.CHART_HEIGHT,
            "width": self.CHART_WIDTH,
        }
        
        if xaxis_title:
            layout["xaxis"] = {
                "title": xaxis_title, 
                "gridcolor": "rgba(255,255,255,0.1)",
                "zerolinecolor": "rgba(255,255,255,0.1)"
            }
        if yaxis_title:
            layout["yaxis"] = {
                "title": yaxis_title, 
                "gridcolor": "rgba(255,255,255,0.1)",
                "zerolinecolor": "rgba(255,255,255,0.1)"
            }
        
        fig.update_layout(**layout)
        return fig
    
    def _save_chart(self, fig, output_dir: Path, filename: str, title: str) -> Dict[str, str]:
        """Save chart and return info dict"""
        chart_path = output_dir / filename
        fig.write_html(str(chart_path), include_plotlyjs='cdn', full_html=True)
        return {
            "name": filename,
            "title": title,
            "path": str(chart_path)
        }
    
    def _generate_pattern_charts(self, patterns: Dict[str, Any], output_dir: Path) -> List[Dict[str, str]]:
        """Generate charts for pattern analysis"""
        charts = []
        
        # Top locations chart - horizontal bar
        if "top_locations" in patterns:
            data = patterns["top_locations"]
            # Sort and take top 10
            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:10]
            locations = [item[0][:25] for item in reversed(sorted_items)]  # Truncate long names
            values = [item[1] for item in reversed(sorted_items)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=values,
                y=locations,
                orientation='h',
                marker=dict(
                    color=values,
                    colorscale='Blues',
                    showscale=False,
                    line=dict(width=0)
                ),
                text=[f"{v:,.0f}" for v in values],
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=10)
            ))
            
            self._apply_layout(fig, "Top 10 Locations by Activity", "Total Count", "")
            fig.update_layout(yaxis=dict(tickfont=dict(size=10)))
            charts.append(self._save_chart(fig, output_dir, "top_locations.html", "Top Locations"))
        
        # Monthly distribution
        if "monthly_distribution" in patterns:
            data = patterns["monthly_distribution"]
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            x_vals = [months[int(k)-1] for k in sorted(data.keys())]
            y_vals = [data[k] for k in sorted(data.keys())]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                marker=dict(color=self.COLORS["primary"]),
                text=[f"{v:,.0f}" for v in y_vals],
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=9)
            ))
            
            self._apply_layout(fig, "Monthly Activity Distribution", "Month", "Count")
            charts.append(self._save_chart(fig, output_dir, "monthly_distribution.html", "Monthly Distribution"))
        
        return charts
    
    def _generate_anomaly_charts(self, anomalies: Dict[str, Any], output_dir: Path) -> List[Dict[str, str]]:
        """Generate charts for anomaly detection"""
        charts = []
        
        # Outlier counts by column
        outlier_counts = anomalies.get("outlier_counts", {})
        if outlier_counts and sum(outlier_counts.values()) > 0:
            columns = list(outlier_counts.keys())
            counts = list(outlier_counts.values())
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=columns,
                y=counts,
                marker=dict(color=self.COLORS["danger"]),
                text=counts,
                textposition='outside',
                textfont=dict(color='#e2e8f0')
            ))
            
            self._apply_layout(fig, "Outliers Detected by Column", "Column", "Outlier Count")
            charts.append(self._save_chart(fig, output_dir, "outlier_counts.html", "Outlier Analysis"))
        
        # High activity anomalies
        high_activity = anomalies.get("high_activity_locations", {})
        if high_activity:
            sorted_items = sorted(high_activity.items(), key=lambda x: x[1], reverse=True)[:8]
            locations = [item[0][:20] for item in reversed(sorted_items)]
            values = [item[1] for item in reversed(sorted_items)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=values,
                y=locations,
                orientation='h',
                marker=dict(color=self.COLORS["warning"]),
                text=[f"{v:,.0f}" for v in values],
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=10)
            ))
            
            self._apply_layout(fig, "High Activity Anomaly Regions", "Activity Level", "")
            charts.append(self._save_chart(fig, output_dir, "high_activity_anomalies.html", "Activity Anomalies"))
        
        return charts
    
    def _generate_classification_charts(self, classification: Dict[str, Any], output_dir: Path) -> List[Dict[str, str]]:
        """Generate charts for classification analysis"""
        charts = []
        
        distribution = classification.get("distribution", {})
        if distribution:
            # Order the levels correctly
            labels = []
            values = []
            colors = []
            
            for i, level in enumerate(self.LEVEL_ORDER):
                if level in distribution:
                    labels.append(level)
                    values.append(distribution[level])
                    colors.append(self.LEVEL_COLORS[i])
            
            # Create donut chart
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='#1a1a2e', width=2)),
                hole=0.5,
                textinfo='label+percent',
                textfont=dict(size=12, color='white'),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
            ))
            
            # Add center text
            total = sum(values)
            fig.add_annotation(
                text=f"<b>{total:,}</b><br>Total",
                x=0.5, y=0.5,
                font=dict(size=16, color='#e2e8f0'),
                showarrow=False
            )
            
            self._apply_layout(fig, "Activity Level Distribution", "", "")
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11)
                )
            )
            charts.append(self._save_chart(fig, output_dir, "classification_distribution.html", "Classification"))
        
        return charts
    
    def _generate_forecast_charts(self, forecasting: Dict[str, Any], output_dir: Path) -> List[Dict[str, str]]:
        """Generate charts for forecasting analysis"""
        charts = []
        
        if forecasting.get("status") != "completed":
            return charts
        
        forecast_data = forecasting.get("forecast", [])
        if not forecast_data:
            return charts
        
        avg = forecasting.get("avg_value", 0)
        trend = forecasting.get("trend", "stable")
        
        # Combine historical average with forecast
        fig = go.Figure()
        
        periods = ["Historical Avg"] + [f"Forecast +{d['period']}" for d in forecast_data]
        values = [avg] + [d['value'] for d in forecast_data]
        colors = [self.COLORS["info"]] + [
            self.COLORS["success"] if d['value'] < avg else self.COLORS["warning"]
            for d in forecast_data
        ]
        
        fig.add_trace(go.Bar(
            x=periods,
            y=values,
            marker=dict(color=colors),
            text=[f"{v:,.0f}" for v in values],
            textposition='outside',
            textfont=dict(color='#e2e8f0')
        ))
        
        self._apply_layout(fig, f"Trend Forecast ({trend.title()})", "Period", "Predicted Value")
        charts.append(self._save_chart(fig, output_dir, "forecast.html", "Forecast"))
        
        return charts
    
    def generate_chart_images(self, results: Dict[str, Any], analysis_id: str, output_dir: str) -> List[str]:
        """Generate static PNG images of charts for PDF export"""
        images = []
        
        try:
            import kaleido  # For static image export
            
            chart_dir = Path(output_dir) / analysis_id
            analyses = results.get("analyses", {})
            
            # Create simplified charts as images
            if "patterns" in analyses and "top_locations" in analyses["patterns"]:
                data = analyses["patterns"]["top_locations"]
                sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:8]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[v for _, v in reversed(sorted_items)],
                    y=[k[:20] for k, _ in reversed(sorted_items)],
                    orientation='h',
                    marker=dict(color='#6366f1')
                ))
                fig.update_layout(
                    title="Top Locations by Activity",
                    width=500, height=300,
                    margin=dict(l=100, r=20, t=40, b=20)
                )
                
                img_path = chart_dir / "top_locations.png"
                fig.write_image(str(img_path))
                images.append(str(img_path))
            
            if "classification" in analyses and "distribution" in analyses["classification"]:
                dist = analyses["classification"]["distribution"]
                
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=list(dist.keys()),
                    values=list(dist.values()),
                    marker=dict(colors=self.LEVEL_COLORS[:len(dist)]),
                    hole=0.4
                ))
                fig.update_layout(
                    title="Classification Distribution",
                    width=400, height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                img_path = chart_dir / "classification.png"
                fig.write_image(str(img_path))
                images.append(str(img_path))
                
        except ImportError:
            # kaleido not installed, skip image generation
            pass
        except Exception as e:
            print(f"Error generating chart images: {e}")
        
        return images
