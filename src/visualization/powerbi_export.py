"""
Power BI Data Export Module
Prepare and export data in optimal format for Power BI dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json


class PowerBIExporter:
    """Export data optimized for Power BI visualization"""
    
    def __init__(self, output_dir: str = "outputs/powerbi"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.time_data: pd.DataFrame = None
        self.month_data: pd.DataFrame = None
        self.severity_data: pd.DataFrame = None
        self.vehicle_data: pd.DataFrame = None
    
    def load_processed_data(self):
        """Load all processed datasets"""
        self.time_data = pd.read_csv("data/processed/accidents_time_processed.csv")
        self.month_data = pd.read_csv("data/processed/accidents_month_processed.csv")
        self.severity_data = pd.read_csv("data/processed/accidents_severity_processed.csv")
        self.vehicle_data = pd.read_csv("data/processed/vehicle_registrations_processed.csv")
        return self
    
    def export_fact_accidents(self) -> str:
        """
        Create fact table for accidents
        Combines time and month data for comprehensive analysis
        """
        # Time-based accidents
        time_df = self.time_data.copy()
        time_df['DataType'] = 'TimeSlot'
        time_df['Period'] = time_df['TimeSlot']
        time_df = time_df.rename(columns={'TimeSlot': 'TimeDimension'})
        
        # Add time period classification
        day_slots = ['06:00-09:00', '09:00-12:00', '12:00-15:00', '15:00-18:00']
        time_df['DayNight'] = time_df['TimeDimension'].apply(lambda x: 'Day' if x in day_slots else 'Night')
        
        # Add hour number for sorting
        time_order = ['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                      '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00']
        time_df['TimeOrder'] = time_df['TimeDimension'].apply(lambda x: time_order.index(x) if x in time_order else 99)
        
        output_path = self.output_dir / 'fact_accidents_time.csv'
        time_df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def export_fact_monthly(self) -> str:
        """
        Create fact table for monthly accidents
        """
        df = self.month_data.copy()
        
        # Add quarter
        quarter_map = {1: 'Q1', 2: 'Q1', 3: 'Q1', 4: 'Q2', 5: 'Q2', 6: 'Q2',
                       7: 'Q3', 8: 'Q3', 9: 'Q3', 10: 'Q4', 11: 'Q4', 12: 'Q4'}
        df['Quarter'] = df['MonthNum'].map(quarter_map)
        
        # Add season
        season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                      6: 'Summer', 7: 'Summer', 8: 'Monsoon', 9: 'Monsoon',
                      10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
        df['Season'] = df['MonthNum'].map(season_map)
        
        output_path = self.output_dir / 'fact_accidents_monthly.csv'
        df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def export_dim_states(self) -> str:
        """
        Create dimension table for states with severity metrics
        """
        df = self.severity_data.copy()
        
        # Filter out totals
        df = df[~df['State'].str.contains('Total', case=False, na=False)]
        
        # Add risk classification
        p75 = df['FatalityRate'].quantile(0.75)
        p50 = df['FatalityRate'].quantile(0.50)
        
        def classify_risk(rate):
            if rate >= p75:
                return 'High Risk'
            elif rate >= p50:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        df['RiskCategory'] = df['FatalityRate'].apply(classify_risk)
        
        # Add region (simplified)
        north = ['Delhi', 'Haryana', 'Punjab', 'Uttar Pradesh', 'Uttarakhand', 'Himachal Pradesh', 
                 'Jammu And Kashmir', 'Ladakh', 'Chandigarh']
        south = ['Tamil Nadu', 'Kerala', 'Karnataka', 'Andhra Pradesh', 'Telangana', 'Puducherry',
                 'Lakshadweep']
        east = ['West Bengal', 'Odisha', 'Bihar', 'Jharkhand', 'Assam', 'Sikkim', 'Meghalaya',
                'Tripura', 'Mizoram', 'Manipur', 'Nagaland', 'Arunachal Pradesh']
        west = ['Maharashtra', 'Gujarat', 'Rajasthan', 'Goa', 'The Dadra And Nagar Haveli And Daman And Diu']
        central = ['Madhya Pradesh', 'Chhattisgarh']
        
        def get_region(state):
            if state in north:
                return 'North'
            elif state in south:
                return 'South'
            elif state in east:
                return 'East'
            elif state in west:
                return 'West'
            elif state in central:
                return 'Central'
            else:
                return 'Other'
        
        df['Region'] = df['State'].apply(get_region)
        
        output_path = self.output_dir / 'dim_states.csv'
        df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def export_summary_kpi(self) -> str:
        """
        Create KPI summary table for Power BI cards
        """
        # Filter out totals for calculations
        severity_clean = self.severity_data[~self.severity_data['State'].str.contains('Total', case=False, na=False)]
        
        kpis = {
            'Total Accidents': int(severity_clean['Cases'].sum()),
            'Total Deaths': int(severity_clean['Died'].sum()),
            'Total Injured': int(severity_clean['Injured'].sum()),
            'Average Fatality Rate': round(severity_clean['FatalityRate'].mean(), 2),
            'Average Injury Rate': round(severity_clean['InjuryRate'].mean(), 2),
            'Highest Fatality State': severity_clean.loc[severity_clean['FatalityRate'].idxmax(), 'State'],
            'Highest Accident State': severity_clean.loc[severity_clean['Cases'].idxmax(), 'State'],
            'Number of States': len(severity_clean),
            'Peak Time Slot': '18:00-21:00',
            'Peak Month': 'May'
        }
        
        kpi_df = pd.DataFrame([kpis])
        
        output_path = self.output_dir / 'summary_kpi.csv'
        kpi_df.to_csv(output_path, index=False)
        
        # Also save as JSON for flexibility
        json_path = self.output_dir / 'summary_kpi.json'
        with open(json_path, 'w') as f:
            json.dump(kpis, f, indent=2)
        
        return str(output_path)
    
    def export_aggregated_views(self) -> List[str]:
        """
        Create pre-aggregated views for faster Power BI loading
        """
        saved_files = []
        
        # 1. State-wise summary
        time_agg = self.time_data.groupby('State')['Accidents'].sum().reset_index()
        time_agg.columns = ['State', 'TotalTimeAccidents']
        
        month_agg = self.month_data.groupby('State')['Accidents'].sum().reset_index()
        month_agg.columns = ['State', 'TotalMonthAccidents']
        
        state_summary = time_agg.merge(month_agg, on='State', how='outer')
        state_summary = state_summary.fillna(0)
        
        output_path = self.output_dir / 'agg_state_summary.csv'
        state_summary.to_csv(output_path, index=False)
        saved_files.append(str(output_path))
        
        # 2. Time slot summary (national)
        time_summary = self.time_data.groupby('TimeSlot').agg({
            'Accidents': ['sum', 'mean', 'std']
        }).reset_index()
        time_summary.columns = ['TimeSlot', 'TotalAccidents', 'AvgAccidents', 'StdAccidents']
        
        time_order = ['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                      '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00']
        time_summary['TimeOrder'] = time_summary['TimeSlot'].apply(lambda x: time_order.index(x))
        time_summary = time_summary.sort_values('TimeOrder')
        
        output_path = self.output_dir / 'agg_time_summary.csv'
        time_summary.to_csv(output_path, index=False)
        saved_files.append(str(output_path))
        
        # 3. Monthly summary (national)
        month_summary = self.month_data.groupby(['Month', 'MonthNum']).agg({
            'Accidents': ['sum', 'mean', 'std']
        }).reset_index()
        month_summary.columns = ['Month', 'MonthNum', 'TotalAccidents', 'AvgAccidents', 'StdAccidents']
        month_summary = month_summary.sort_values('MonthNum')
        
        output_path = self.output_dir / 'agg_month_summary.csv'
        month_summary.to_csv(output_path, index=False)
        saved_files.append(str(output_path))
        
        return saved_files
    
    def export_all(self) -> Dict[str, List[str]]:
        """
        Export all Power BI ready files
        """
        print("Exporting data for Power BI...")
        
        exports = {
            'fact_tables': [],
            'dimension_tables': [],
            'kpi_tables': [],
            'aggregated_views': []
        }
        
        # Fact tables
        exports['fact_tables'].append(self.export_fact_accidents())
        print("  ✓ Fact: Accidents by Time")
        
        exports['fact_tables'].append(self.export_fact_monthly())
        print("  ✓ Fact: Accidents by Month")
        
        # Dimension tables
        exports['dimension_tables'].append(self.export_dim_states())
        print("  ✓ Dim: States")
        
        # KPI summaries
        exports['kpi_tables'].append(self.export_summary_kpi())
        print("  ✓ KPI: Summary")
        
        # Aggregated views
        exports['aggregated_views'] = self.export_aggregated_views()
        print("  ✓ Aggregated Views")
        
        total_files = sum(len(v) for v in exports.values())
        print(f"\nExported {total_files} files to {self.output_dir}")
        
        return exports


# Quick test when run directly
if __name__ == "__main__":
    exporter = PowerBIExporter()
    exporter.load_processed_data()
    result = exporter.export_all()
    
    print("\nExported files:")
    for category, files in result.items():
        print(f"\n{category}:")
        for f in files:
            print(f"  - {f}")
