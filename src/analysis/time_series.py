"""
Time Series Analysis Module
Temporal trend analysis, seasonality detection, and forecasting preparation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrendResult:
    """Results from trend analysis"""
    direction: str  # 'increasing', 'decreasing', 'stable'
    slope: float
    r_squared: float


@dataclass
class SeasonalityResult:
    """Results from seasonality detection"""
    has_seasonality: bool
    peak_months: List[str]
    trough_months: List[str]
    amplitude: float


class TimeSeriesAnalyzer:
    """Analyze temporal patterns in traffic data"""
    
    def __init__(self):
        self.month_data: Optional[pd.DataFrame] = None
        self.time_data: Optional[pd.DataFrame] = None
    
    def load_processed_data(self,
                            month_file: str = "data/processed/accidents_month_processed.csv",
                            time_file: str = "data/processed/accidents_time_processed.csv"):
        """Load processed datasets"""
        self.month_data = pd.read_csv(month_file)
        self.time_data = pd.read_csv(time_file)
        return self
    
    def analyze_monthly_trend(self, state: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze monthly trend for a state or nationally
        """
        df = self.month_data.copy()
        
        if state:
            df = df[df['State'] == state]
        
        # Aggregate by month
        monthly = df.groupby(['Month', 'MonthNum']).agg({
            'Accidents': 'sum'
        }).reset_index().sort_values('MonthNum')
        
        # Calculate month-over-month change
        monthly['MoM_Change'] = monthly['Accidents'].diff()
        monthly['MoM_Percent'] = (monthly['MoM_Change'] / monthly['Accidents'].shift(1) * 100).round(2)
        
        # Calculate cumulative accidents
        monthly['Cumulative'] = monthly['Accidents'].cumsum()
        
        # Calculate 3-month moving average
        monthly['MA_3'] = monthly['Accidents'].rolling(window=3, center=True).mean().round(0)
        
        return monthly
    
    def detect_seasonality(self, state: Optional[str] = None) -> Dict:
        """
        Detect seasonal patterns in monthly data
        """
        df = self.month_data.copy()
        
        if state:
            df = df[df['State'] == state]
        
        # Aggregate by month
        monthly = df.groupby(['Month', 'MonthNum']).agg({
            'Accidents': 'sum'
        }).reset_index().sort_values('MonthNum')
        
        mean_accidents = monthly['Accidents'].mean()
        std_accidents = monthly['Accidents'].std()
        
        # Identify peak and trough months
        monthly['ZScore'] = (monthly['Accidents'] - mean_accidents) / std_accidents
        
        peak_months = monthly[monthly['ZScore'] > 1]['Month'].tolist()
        trough_months = monthly[monthly['ZScore'] < -1]['Month'].tolist()
        
        # Calculate seasonality amplitude
        amplitude = (monthly['Accidents'].max() - monthly['Accidents'].min()) / mean_accidents * 100
        
        # Check if seasonality exists (amplitude > 20% indicates seasonality)
        has_seasonality = amplitude > 20
        
        return {
            'has_seasonality': has_seasonality,
            'amplitude': round(amplitude, 2),
            'peak_months': peak_months,
            'trough_months': trough_months,
            'max_month': monthly.loc[monthly['Accidents'].idxmax(), 'Month'],
            'min_month': monthly.loc[monthly['Accidents'].idxmin(), 'Month'],
            'coefficient_of_variation': round(std_accidents / mean_accidents * 100, 2)
        }
    
    def analyze_hourly_distribution(self, state: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze distribution of accidents across time slots
        """
        df = self.time_data.copy()
        
        if state:
            df = df[df['State'] == state]
        
        # Aggregate by time slot
        hourly = df.groupby('TimeSlot').agg({
            'Accidents': 'sum'
        }).reset_index()
        
        total = hourly['Accidents'].sum()
        hourly['Percentage'] = (hourly['Accidents'] / total * 100).round(2)
        
        # Sort by time slot (natural order)
        time_order = ['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                      '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00']
        hourly['TimeOrder'] = hourly['TimeSlot'].apply(lambda x: time_order.index(x) if x in time_order else 99)
        hourly = hourly.sort_values('TimeOrder')
        
        # Calculate cumulative percentage
        hourly['CumulativePercent'] = hourly['Percentage'].cumsum().round(2)
        
        return hourly
    
    def calculate_peak_to_offpeak_ratio(self, state: Optional[str] = None) -> Dict:
        """
        Calculate ratio between peak and off-peak accidents
        """
        hourly = self.analyze_hourly_distribution(state)
        
        # Define peak hours (18:00-21:00, 15:00-18:00) and off-peak (00:00-06:00)
        peak_slots = ['15:00-18:00', '18:00-21:00']
        offpeak_slots = ['00:00-03:00', '03:00-06:00']
        
        peak_accidents = hourly[hourly['TimeSlot'].isin(peak_slots)]['Accidents'].sum()
        offpeak_accidents = hourly[hourly['TimeSlot'].isin(offpeak_slots)]['Accidents'].sum()
        
        ratio = peak_accidents / offpeak_accidents if offpeak_accidents > 0 else 0
        
        return {
            'peak_accidents': peak_accidents,
            'offpeak_accidents': offpeak_accidents,
            'peak_to_offpeak_ratio': round(ratio, 2),
            'peak_percentage': round(peak_accidents / hourly['Accidents'].sum() * 100, 2),
            'offpeak_percentage': round(offpeak_accidents / hourly['Accidents'].sum() * 100, 2)
        }
    
    def compare_quarters(self, state: Optional[str] = None) -> pd.DataFrame:
        """
        Compare accident patterns across quarters
        """
        df = self.month_data.copy()
        
        if state:
            df = df[df['State'] == state]
        
        # Assign quarters
        quarter_map = {
            1: 'Q1', 2: 'Q1', 3: 'Q1',
            4: 'Q2', 5: 'Q2', 6: 'Q2',
            7: 'Q3', 8: 'Q3', 9: 'Q3',
            10: 'Q4', 11: 'Q4', 12: 'Q4'
        }
        df['Quarter'] = df['MonthNum'].map(quarter_map)
        
        # Aggregate by quarter
        quarterly = df.groupby('Quarter').agg({
            'Accidents': 'sum'
        }).reset_index()
        
        # Sort quarters
        quarter_order = ['Q1', 'Q2', 'Q3', 'Q4']
        quarterly['QuarterOrder'] = quarterly['Quarter'].apply(lambda x: quarter_order.index(x))
        quarterly = quarterly.sort_values('QuarterOrder')
        
        # Calculate percentage
        total = quarterly['Accidents'].sum()
        quarterly['Percentage'] = (quarterly['Accidents'] / total * 100).round(2)
        
        # Quarter-over-quarter change
        quarterly['QoQ_Change'] = quarterly['Accidents'].diff()
        quarterly['QoQ_Percent'] = (quarterly['QoQ_Change'] / quarterly['Accidents'].shift(1) * 100).round(2)
        
        return quarterly
    
    def get_state_seasonality_comparison(self) -> pd.DataFrame:
        """
        Compare seasonality across all states
        """
        states = self.month_data['State'].unique()
        
        results = []
        for state in states:
            seasonality = self.detect_seasonality(state)
            peak_ratio = self.calculate_peak_to_offpeak_ratio(state)
            
            results.append({
                'State': state,
                'HasSeasonality': seasonality['has_seasonality'],
                'Amplitude': seasonality['amplitude'],
                'PeakMonth': seasonality['max_month'],
                'TroughMonth': seasonality['min_month'],
                'PeakToOffpeakRatio': peak_ratio['peak_to_offpeak_ratio']
            })
        
        return pd.DataFrame(results).sort_values('Amplitude', ascending=False)
    
    def generate_timeseries_summary(self) -> Dict:
        """
        Generate comprehensive time series summary
        """
        national_seasonality = self.detect_seasonality()
        national_peak_ratio = self.calculate_peak_to_offpeak_ratio()
        quarterly = self.compare_quarters()
        state_comparison = self.get_state_seasonality_comparison()
        
        return {
            'national_seasonality': national_seasonality,
            'peak_to_offpeak_ratio': national_peak_ratio,
            'highest_quarter': quarterly.loc[quarterly['Accidents'].idxmax(), 'Quarter'],
            'lowest_quarter': quarterly.loc[quarterly['Accidents'].idxmin(), 'Quarter'],
            'states_with_high_seasonality': state_comparison[state_comparison['Amplitude'] > 30]['State'].tolist(),
            'avg_peak_to_offpeak_ratio': state_comparison['PeakToOffpeakRatio'].mean().round(2)
        }


# Quick test when run directly
if __name__ == "__main__":
    analyzer = TimeSeriesAnalyzer()
    analyzer.load_processed_data()
    
    print("=" * 60)
    print("MONTHLY TREND ANALYSIS (National)")
    print("=" * 60)
    print(analyzer.analyze_monthly_trend().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SEASONALITY DETECTION")
    print("=" * 60)
    seasonality = analyzer.detect_seasonality()
    for key, value in seasonality.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("HOURLY DISTRIBUTION")
    print("=" * 60)
    print(analyzer.analyze_hourly_distribution().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("QUARTERLY COMPARISON")
    print("=" * 60)
    print(analyzer.compare_quarters().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TIME SERIES SUMMARY")
    print("=" * 60)
    summary = analyzer.generate_timeseries_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
