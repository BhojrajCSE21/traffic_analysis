"""
Pattern Recognition Module
Identifies traffic patterns including peak hours, seasonal trends, and correlations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PeakHourResult:
    """Results from peak hour analysis"""
    time_slot: str
    total_accidents: int
    percentage: float
    classification: str  # 'peak', 'moderate', 'low'


@dataclass
class SeasonalPattern:
    """Seasonal pattern results"""
    month: str
    total_accidents: int
    deviation_from_mean: float
    classification: str  # 'high', 'normal', 'low'


class PatternRecognizer:
    """Analyze traffic patterns from accident data"""
    
    def __init__(self):
        self.time_data: Optional[pd.DataFrame] = None
        self.month_data: Optional[pd.DataFrame] = None
    
    def load_processed_data(self, time_file: str = "data/processed/accidents_time_processed.csv",
                            month_file: str = "data/processed/accidents_month_processed.csv"):
        """Load processed datasets"""
        self.time_data = pd.read_csv(time_file)
        self.month_data = pd.read_csv(month_file)
        return self
    
    def analyze_peak_hours(self, state: Optional[str] = None) -> pd.DataFrame:
        """
        Identify peak accident hours
        Returns DataFrame with time slots ranked by accident frequency
        """
        df = self.time_data.copy()
        
        if state:
            df = df[df['State'] == state]
        
        # Aggregate by time slot
        time_analysis = df.groupby('TimeSlot').agg({
            'Accidents': ['sum', 'mean', 'std']
        }).reset_index()
        time_analysis.columns = ['TimeSlot', 'TotalAccidents', 'MeanAccidents', 'StdAccidents']
        
        # Calculate percentage
        total = time_analysis['TotalAccidents'].sum()
        time_analysis['Percentage'] = (time_analysis['TotalAccidents'] / total * 100).round(2)
        
        # Classify time slots
        mean_pct = time_analysis['Percentage'].mean()
        std_pct = time_analysis['Percentage'].std()
        
        def classify(pct):
            if pct > mean_pct + std_pct:
                return 'PEAK'
            elif pct < mean_pct - std_pct:
                return 'LOW'
            else:
                return 'MODERATE'
        
        time_analysis['Classification'] = time_analysis['Percentage'].apply(classify)
        
        # Sort by accidents descending
        time_analysis = time_analysis.sort_values('TotalAccidents', ascending=False)
        
        # Add rank
        time_analysis['Rank'] = range(1, len(time_analysis) + 1)
        
        return time_analysis
    
    def analyze_monthly_patterns(self, state: Optional[str] = None) -> pd.DataFrame:
        """
        Identify monthly/seasonal patterns
        Returns DataFrame with months ranked by accident frequency
        """
        df = self.month_data.copy()
        
        if state:
            df = df[df['State'] == state]
        
        # Aggregate by month
        month_analysis = df.groupby(['Month', 'MonthNum']).agg({
            'Accidents': ['sum', 'mean', 'std']
        }).reset_index()
        month_analysis.columns = ['Month', 'MonthNum', 'TotalAccidents', 'MeanAccidents', 'StdAccidents']
        
        # Calculate percentage
        total = month_analysis['TotalAccidents'].sum()
        month_analysis['Percentage'] = (month_analysis['TotalAccidents'] / total * 100).round(2)
        
        # Calculate deviation from mean
        mean_accidents = month_analysis['TotalAccidents'].mean()
        month_analysis['DeviationFromMean'] = ((month_analysis['TotalAccidents'] - mean_accidents) / mean_accidents * 100).round(2)
        
        # Classify months
        def classify(deviation):
            if deviation > 10:
                return 'HIGH'
            elif deviation < -10:
                return 'LOW'
            else:
                return 'NORMAL'
        
        month_analysis['Classification'] = month_analysis['DeviationFromMean'].apply(classify)
        
        # Sort by month number
        month_analysis = month_analysis.sort_values('MonthNum')
        
        return month_analysis
    
    def get_state_comparison(self) -> pd.DataFrame:
        """
        Compare accident patterns across states
        Returns DataFrame with state-wise statistics
        """
        df = self.time_data.copy()
        
        state_analysis = df.groupby('State').agg({
            'Accidents': ['sum', 'mean', 'std']
        }).reset_index()
        state_analysis.columns = ['State', 'TotalAccidents', 'MeanAccidents', 'StdAccidents']
        
        # Calculate percentage of national total
        total = state_analysis['TotalAccidents'].sum()
        state_analysis['NationalShare'] = (state_analysis['TotalAccidents'] / total * 100).round(2)
        
        # Rank by accidents
        state_analysis = state_analysis.sort_values('TotalAccidents', ascending=False)
        state_analysis['Rank'] = range(1, len(state_analysis) + 1)
        
        return state_analysis
    
    def get_peak_hour_by_state(self) -> pd.DataFrame:
        """
        Find the peak hour for each state
        """
        df = self.time_data.copy()
        
        # Find peak hour for each state
        peak_hours = df.loc[df.groupby('State')['Accidents'].idxmax()]
        peak_hours = peak_hours[['State', 'TimeSlot', 'Accidents']]
        peak_hours.columns = ['State', 'PeakTimeSlot', 'PeakAccidents']
        
        return peak_hours.sort_values('PeakAccidents', ascending=False)
    
    def get_day_vs_night_comparison(self) -> pd.DataFrame:
        """
        Compare day vs night accident patterns
        Day: 06:00-18:00, Night: 18:00-06:00
        """
        df = self.time_data.copy()
        
        # Classify as day or night
        day_slots = ['06:00-09:00', '09:00-12:00', '12:00-15:00', '15:00-18:00']
        df['Period'] = df['TimeSlot'].apply(lambda x: 'Day' if x in day_slots else 'Night')
        
        # Aggregate
        period_analysis = df.groupby(['State', 'Period']).agg({
            'Accidents': 'sum'
        }).reset_index()
        
        # Pivot for easy comparison
        pivot = period_analysis.pivot(index='State', columns='Period', values='Accidents').reset_index()
        pivot['DayNightRatio'] = (pivot['Day'] / pivot['Night']).round(2)
        pivot['Total'] = pivot['Day'] + pivot['Night']
        pivot['DayPercentage'] = (pivot['Day'] / pivot['Total'] * 100).round(2)
        
        return pivot.sort_values('Total', ascending=False)
    
    def generate_pattern_summary(self) -> Dict:
        """
        Generate comprehensive pattern summary
        """
        peak_hours = self.analyze_peak_hours()
        monthly = self.analyze_monthly_patterns()
        state_comparison = self.get_state_comparison()
        day_night = self.get_day_vs_night_comparison()
        
        return {
            'peak_time_slots': peak_hours[peak_hours['Classification'] == 'PEAK']['TimeSlot'].tolist(),
            'safest_time_slots': peak_hours[peak_hours['Classification'] == 'LOW']['TimeSlot'].tolist(),
            'high_accident_months': monthly[monthly['Classification'] == 'HIGH']['Month'].tolist(),
            'low_accident_months': monthly[monthly['Classification'] == 'LOW']['Month'].tolist(),
            'top_5_states': state_comparison.head(5)['State'].tolist(),
            'national_day_percentage': day_night['DayPercentage'].mean().round(2),
            'total_accidents_analyzed': self.time_data['Accidents'].sum()
        }


# Quick test when run directly
if __name__ == "__main__":
    analyzer = PatternRecognizer()
    analyzer.load_processed_data()
    
    print("=" * 60)
    print("PEAK HOUR ANALYSIS (National)")
    print("=" * 60)
    print(analyzer.analyze_peak_hours().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("MONTHLY PATTERN ANALYSIS")
    print("=" * 60)
    print(analyzer.analyze_monthly_patterns().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TOP 10 STATES BY ACCIDENTS")
    print("=" * 60)
    print(analyzer.get_state_comparison().head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("PATTERN SUMMARY")
    print("=" * 60)
    summary = analyzer.generate_pattern_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
