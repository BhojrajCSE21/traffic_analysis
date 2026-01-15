"""
Visualization Plots Module
Generate static and interactive visualizations for traffic analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class TrafficVisualizer:
    """Generate visualizations for traffic analysis"""
    
    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.time_data: Optional[pd.DataFrame] = None
        self.month_data: Optional[pd.DataFrame] = None
        self.severity_data: Optional[pd.DataFrame] = None
        self.vehicle_data: Optional[pd.DataFrame] = None
    
    def load_processed_data(self,
                            time_file: str = "data/processed/accidents_time_processed.csv",
                            month_file: str = "data/processed/accidents_month_processed.csv",
                            severity_file: str = "data/processed/accidents_severity_processed.csv",
                            vehicle_file: str = "data/processed/vehicle_registrations_processed.csv"):
        """Load processed datasets"""
        self.time_data = pd.read_csv(time_file)
        self.month_data = pd.read_csv(month_file)
        self.severity_data = pd.read_csv(severity_file)
        self.vehicle_data = pd.read_csv(vehicle_file)
        return self
    
    def plot_accidents_by_time(self, save: bool = True) -> go.Figure:
        """
        Create bar chart of accidents by time slot
        """
        df = self.time_data.groupby('TimeSlot')['Accidents'].sum().reset_index()
        
        # Order time slots
        time_order = ['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                      '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00']
        df['TimeOrder'] = df['TimeSlot'].apply(lambda x: time_order.index(x) if x in time_order else 99)
        df = df.sort_values('TimeOrder')
        
        # Color based on accident volume
        colors = ['#2ecc71' if x < df['Accidents'].quantile(0.25) else 
                  '#f39c12' if x < df['Accidents'].quantile(0.75) else '#e74c3c' 
                  for x in df['Accidents']]
        
        fig = go.Figure(data=[
            go.Bar(x=df['TimeSlot'], y=df['Accidents'], marker_color=colors,
                   text=df['Accidents'].apply(lambda x: f'{x:,.0f}'),
                   textposition='outside')
        ])
        
        fig.update_layout(
            title='Traffic Accidents by Time of Day (2023)',
            xaxis_title='Time Slot',
            yaxis_title='Number of Accidents',
            template='plotly_white',
            height=500
        )
        
        if save:
            fig.write_html(self.output_dir / 'accidents_by_time.html')
            fig.write_image(self.output_dir / 'accidents_by_time.png', scale=2)
        
        return fig
    
    def plot_accidents_by_month(self, save: bool = True) -> go.Figure:
        """
        Create line chart of accidents by month
        """
        df = self.month_data.groupby(['Month', 'MonthNum'])['Accidents'].sum().reset_index()
        df = df.sort_values('MonthNum')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Month'], y=df['Accidents'],
            mode='lines+markers+text',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            text=df['Accidents'].apply(lambda x: f'{x/1000:.0f}K'),
            textposition='top center'
        ))
        
        fig.update_layout(
            title='Monthly Traffic Accidents Trend (2023)',
            xaxis_title='Month',
            yaxis_title='Number of Accidents',
            template='plotly_white',
            height=500
        )
        
        if save:
            fig.write_html(self.output_dir / 'accidents_by_month.html')
            fig.write_image(self.output_dir / 'accidents_by_month.png', scale=2)
        
        return fig
    
    def plot_top_states_severity(self, top_n: int = 15, save: bool = True) -> go.Figure:
        """
        Create horizontal bar chart of top states by accident cases
        """
        # Filter out total rows
        df = self.severity_data[~self.severity_data['State'].str.contains('Total', case=False, na=False)]
        df = df.nlargest(top_n, 'Cases')
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Total Cases', 'Fatality Rate (%)'))
        
        # Cases bar
        fig.add_trace(
            go.Bar(y=df['State'], x=df['Cases'], orientation='h',
                   marker_color='#3498db', name='Cases'),
            row=1, col=1
        )
        
        # Fatality rate bar
        fig.add_trace(
            go.Bar(y=df['State'], x=df['FatalityRate'], orientation='h',
                   marker_color='#e74c3c', name='Fatality Rate'),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Top {top_n} States: Accidents & Fatality Rate',
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        if save:
            fig.write_html(self.output_dir / 'top_states_severity.html')
            fig.write_image(self.output_dir / 'top_states_severity.png', scale=2)
        
        return fig
    
    def plot_day_night_comparison(self, save: bool = True) -> go.Figure:
        """
        Create pie chart comparing day vs night accidents
        """
        df = self.time_data.copy()
        
        day_slots = ['06:00-09:00', '09:00-12:00', '12:00-15:00', '15:00-18:00']
        df['Period'] = df['TimeSlot'].apply(lambda x: 'Day (6AM-6PM)' if x in day_slots else 'Night (6PM-6AM)')
        
        period_totals = df.groupby('Period')['Accidents'].sum().reset_index()
        
        fig = go.Figure(data=[go.Pie(
            labels=period_totals['Period'],
            values=period_totals['Accidents'],
            hole=0.4,
            marker_colors=['#f39c12', '#2c3e50'],
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig.update_layout(
            title='Day vs Night Accident Distribution',
            template='plotly_white',
            height=500
        )
        
        if save:
            fig.write_html(self.output_dir / 'day_night_comparison.html')
            fig.write_image(self.output_dir / 'day_night_comparison.png', scale=2)
        
        return fig
    
    def plot_heatmap_state_time(self, top_n: int = 15, save: bool = True) -> go.Figure:
        """
        Create heatmap of accidents by state and time slot
        """
        # Filter out total rows
        df = self.time_data[~self.time_data['State'].str.contains('Total', case=False, na=False)]
        
        # Get top states
        top_states = df.groupby('State')['Accidents'].sum().nlargest(top_n).index
        df = df[df['State'].isin(top_states)]
        
        # Pivot for heatmap
        pivot = df.pivot_table(index='State', columns='TimeSlot', values='Accidents', aggfunc='sum')
        
        # Order columns
        time_order = ['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                      '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00']
        pivot = pivot[[c for c in time_order if c in pivot.columns]]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn_r',
            text=pivot.values.astype(int),
            texttemplate='%{text:,}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=f'Accidents Heatmap: Top {top_n} States by Time Slot',
            xaxis_title='Time Slot',
            yaxis_title='State',
            template='plotly_white',
            height=700
        )
        
        if save:
            fig.write_html(self.output_dir / 'heatmap_state_time.html')
            fig.write_image(self.output_dir / 'heatmap_state_time.png', scale=2)
        
        return fig
    
    def plot_fatality_vs_cases(self, save: bool = True) -> go.Figure:
        """
        Create scatter plot of fatality rate vs total cases
        """
        # Filter out total rows
        df = self.severity_data[~self.severity_data['State'].str.contains('Total', case=False, na=False)]
        
        fig = px.scatter(
            df, x='Cases', y='FatalityRate',
            size='Died', color='FatalityRate',
            hover_name='State',
            color_continuous_scale='RdYlGn_r',
            size_max=50
        )
        
        # Add average lines
        avg_cases = df['Cases'].mean()
        avg_fatality = df['FatalityRate'].mean()
        
        fig.add_hline(y=avg_fatality, line_dash="dash", line_color="gray",
                      annotation_text=f"Avg Fatality Rate: {avg_fatality:.1f}%")
        fig.add_vline(x=avg_cases, line_dash="dash", line_color="gray",
                      annotation_text=f"Avg Cases: {avg_cases:.0f}")
        
        fig.update_layout(
            title='State-wise: Fatality Rate vs Total Accidents',
            xaxis_title='Total Accident Cases',
            yaxis_title='Fatality Rate (%)',
            template='plotly_white',
            height=600
        )
        
        if save:
            fig.write_html(self.output_dir / 'fatality_vs_cases.html')
            fig.write_image(self.output_dir / 'fatality_vs_cases.png', scale=2)
        
        return fig
    
    def plot_quarterly_trend(self, save: bool = True) -> go.Figure:
        """
        Create quarterly comparison chart
        """
        df = self.month_data.copy()
        
        quarter_map = {1: 'Q1', 2: 'Q1', 3: 'Q1', 4: 'Q2', 5: 'Q2', 6: 'Q2',
                       7: 'Q3', 8: 'Q3', 9: 'Q3', 10: 'Q4', 11: 'Q4', 12: 'Q4'}
        df['Quarter'] = df['MonthNum'].map(quarter_map)
        
        quarterly = df.groupby('Quarter')['Accidents'].sum().reset_index()
        quarterly = quarterly.sort_values('Quarter')
        
        fig = go.Figure(data=[
            go.Bar(x=quarterly['Quarter'], y=quarterly['Accidents'],
                   marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                   text=quarterly['Accidents'].apply(lambda x: f'{x/1000:.0f}K'),
                   textposition='outside')
        ])
        
        fig.update_layout(
            title='Quarterly Traffic Accidents (2023)',
            xaxis_title='Quarter',
            yaxis_title='Number of Accidents',
            template='plotly_white',
            height=500
        )
        
        if save:
            fig.write_html(self.output_dir / 'quarterly_trend.html')
            fig.write_image(self.output_dir / 'quarterly_trend.png', scale=2)
        
        return fig
    
    def generate_all_visualizations(self) -> List[str]:
        """
        Generate all visualizations and return list of saved files
        """
        print("Generating visualizations...")
        saved_files = []
        
        self.plot_accidents_by_time()
        saved_files.append('accidents_by_time.png')
        print("  ✓ Accidents by Time")
        
        self.plot_accidents_by_month()
        saved_files.append('accidents_by_month.png')
        print("  ✓ Accidents by Month")
        
        self.plot_top_states_severity()
        saved_files.append('top_states_severity.png')
        print("  ✓ Top States Severity")
        
        self.plot_day_night_comparison()
        saved_files.append('day_night_comparison.png')
        print("  ✓ Day/Night Comparison")
        
        self.plot_heatmap_state_time()
        saved_files.append('heatmap_state_time.png')
        print("  ✓ State-Time Heatmap")
        
        self.plot_fatality_vs_cases()
        saved_files.append('fatality_vs_cases.png')
        print("  ✓ Fatality vs Cases Scatter")
        
        self.plot_quarterly_trend()
        saved_files.append('quarterly_trend.png')
        print("  ✓ Quarterly Trend")
        
        print(f"\nGenerated {len(saved_files)} visualizations in {self.output_dir}")
        return saved_files


# Quick test when run directly
if __name__ == "__main__":
    visualizer = TrafficVisualizer()
    visualizer.load_processed_data()
    saved = visualizer.generate_all_visualizations()
    print("\nSaved files:", saved)
