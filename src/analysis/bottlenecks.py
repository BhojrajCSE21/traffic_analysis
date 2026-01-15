"""
Bottleneck Identification Module
Identifies chronic congestion points, high-risk areas, and severity hotspots
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BottleneckResult:
    """Results from bottleneck analysis"""
    location: str
    severity_score: float
    fatality_rate: float
    injury_rate: float
    classification: str  # 'critical', 'severe', 'moderate', 'low'


class BottleneckIdentifier:
    """Identify traffic bottlenecks and high-risk areas"""
    
    def __init__(self):
        self.severity_data: Optional[pd.DataFrame] = None
        self.vehicle_data: Optional[pd.DataFrame] = None
        self.time_data: Optional[pd.DataFrame] = None
    
    def load_processed_data(self, 
                            severity_file: str = "data/processed/accidents_severity_processed.csv",
                            vehicle_file: str = "data/processed/vehicle_registrations_processed.csv",
                            time_file: str = "data/processed/accidents_time_processed.csv"):
        """Load processed datasets"""
        self.severity_data = pd.read_csv(severity_file)
        self.vehicle_data = pd.read_csv(vehicle_file)
        self.time_data = pd.read_csv(time_file)
        return self
    
    def analyze_severity_hotspots(self) -> pd.DataFrame:
        """
        Identify states/cities with highest accident severity
        Based on fatality rate and injury rate
        """
        df = self.severity_data.copy()
        
        # Calculate severity score (weighted combination of fatality and injury rates)
        # Higher weight on fatality rate as it's more critical
        df['SeverityScore'] = (df['FatalityRate'] * 0.7 + df['InjuryRate'] * 0.3).round(2)
        
        # Classify severity
        p75 = df['SeverityScore'].quantile(0.75)
        p50 = df['SeverityScore'].quantile(0.50)
        p25 = df['SeverityScore'].quantile(0.25)
        
        def classify_severity(score):
            if score >= p75:
                return 'CRITICAL'
            elif score >= p50:
                return 'SEVERE'
            elif score >= p25:
                return 'MODERATE'
            else:
                return 'LOW'
        
        df['SeverityClassification'] = df['SeverityScore'].apply(classify_severity)
        
        # Sort by severity score
        df = df.sort_values('SeverityScore', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def analyze_fatality_hotspots(self) -> pd.DataFrame:
        """
        Identify states with highest fatality rates
        """
        df = self.severity_data.copy()
        
        # Sort by fatality rate
        df = df.sort_values('FatalityRate', ascending=False)
        
        # Add national average comparison
        national_avg = df['FatalityRate'].mean()
        df['AboveNationalAvg'] = df['FatalityRate'] > national_avg
        df['DeviationFromAvg'] = ((df['FatalityRate'] - national_avg) / national_avg * 100).round(2)
        
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def analyze_accident_volume_hotspots(self) -> pd.DataFrame:
        """
        Identify states with highest total accidents
        """
        df = self.severity_data.copy()
        
        # Sort by total cases
        df = df.sort_values('Cases', ascending=False)
        
        # Calculate percentage of national total
        total = df['Cases'].sum()
        df['NationalShare'] = (df['Cases'] / total * 100).round(2)
        
        # Cumulative percentage
        df['CumulativeShare'] = df['NationalShare'].cumsum().round(2)
        
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def analyze_time_concentration(self) -> pd.DataFrame:
        """
        Identify states where accidents are highly concentrated in specific time slots
        Higher concentration = potential bottleneck during those hours
        """
        df = self.time_data.copy()
        
        # Calculate Herfindahl index for time concentration
        def calc_concentration(group):
            total = group['Accidents'].sum()
            if total == 0:
                return 0
            shares = (group['Accidents'] / total) ** 2
            return shares.sum()
        
        concentration = df.groupby('State').apply(calc_concentration, include_groups=False).reset_index()
        concentration.columns = ['State', 'TimeConcentration']
        
        # Higher concentration index means accidents are concentrated in fewer time slots
        concentration = concentration.sort_values('TimeConcentration', ascending=False)
        
        # Classify
        mean_conc = concentration['TimeConcentration'].mean()
        concentration['HighConcentration'] = concentration['TimeConcentration'] > mean_conc
        
        return concentration
    
    def analyze_vehicle_density_risk(self) -> pd.DataFrame:
        """
        Analyze cities by vehicle density
        Higher density may correlate with congestion bottlenecks
        """
        df = self.vehicle_data.copy()
        
        # Calculate total vehicles per city
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df['TotalVehicles'] = df[numeric_cols].sum(axis=1)
        
        # Calculate vehicle mix (two wheelers vs four wheelers ratio)
        if 'TotalTwoWheelers' in df.columns and 'TotalFourWheelers' in df.columns:
            df['TwoWheelerRatio'] = (df['TotalTwoWheelers'] / df['TotalVehicles'] * 100).round(2)
        
        # Sort by total vehicles
        df = df.sort_values('TotalVehicles', ascending=False)
        
        # Classify as high/medium/low density
        p75 = df['TotalVehicles'].quantile(0.75)
        p25 = df['TotalVehicles'].quantile(0.25)
        
        def classify_density(total):
            if total >= p75:
                return 'HIGH'
            elif total >= p25:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        df['DensityClass'] = df['TotalVehicles'].apply(classify_density)
        
        return df[['City', 'TotalVehicles', 'DensityClass']].head(20)
    
    def identify_critical_bottlenecks(self) -> pd.DataFrame:
        """
        Combine multiple factors to identify critical bottlenecks
        """
        severity = self.analyze_severity_hotspots()[['State', 'SeverityScore', 'SeverityClassification']]
        volume = self.analyze_accident_volume_hotspots()[['State', 'Cases', 'NationalShare']]
        concentration = self.analyze_time_concentration()
        
        # Merge all factors
        combined = severity.merge(volume, on='State')
        combined = combined.merge(concentration, on='State')
        
        # Calculate overall risk score
        # Normalize each factor to 0-100 scale
        combined['NormSeverity'] = (combined['SeverityScore'] / combined['SeverityScore'].max() * 100).round(2)
        combined['NormVolume'] = (combined['Cases'] / combined['Cases'].max() * 100).round(2)
        combined['NormConcentration'] = (combined['TimeConcentration'] / combined['TimeConcentration'].max() * 100).round(2)
        
        # Weighted risk score
        combined['OverallRiskScore'] = (
            combined['NormSeverity'] * 0.4 + 
            combined['NormVolume'] * 0.4 + 
            combined['NormConcentration'] * 0.2
        ).round(2)
        
        # Classify overall risk
        p75 = combined['OverallRiskScore'].quantile(0.75)
        p50 = combined['OverallRiskScore'].quantile(0.50)
        
        def classify_risk(score):
            if score >= p75:
                return 'CRITICAL'
            elif score >= p50:
                return 'HIGH'
            else:
                return 'MODERATE'
        
        combined['RiskClassification'] = combined['OverallRiskScore'].apply(classify_risk)
        
        # Sort by risk score
        combined = combined.sort_values('OverallRiskScore', ascending=False)
        combined['Rank'] = range(1, len(combined) + 1)
        
        return combined
    
    def generate_bottleneck_summary(self) -> Dict:
        """
        Generate comprehensive bottleneck summary
        """
        critical = self.identify_critical_bottlenecks()
        fatality = self.analyze_fatality_hotspots()
        
        return {
            'critical_states': critical[critical['RiskClassification'] == 'CRITICAL']['State'].tolist(),
            'high_risk_states': critical[critical['RiskClassification'] == 'HIGH']['State'].tolist(),
            'top_5_by_severity': fatality.head(5)['State'].tolist(),
            'top_5_by_volume': critical.nsmallest(5, 'Rank')['State'].tolist(),
            'national_fatality_rate': fatality['FatalityRate'].mean().round(2),
            'national_injury_rate': self.severity_data['InjuryRate'].mean().round(2)
        }


# Quick test when run directly
if __name__ == "__main__":
    identifier = BottleneckIdentifier()
    identifier.load_processed_data()
    
    print("=" * 60)
    print("SEVERITY HOTSPOTS (Top 10)")
    print("=" * 60)
    print(identifier.analyze_severity_hotspots().head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("FATALITY HOTSPOTS (Top 10)")
    print("=" * 60)
    print(identifier.analyze_fatality_hotspots().head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("CRITICAL BOTTLENECKS (Top 10)")
    print("=" * 60)
    critical = identifier.identify_critical_bottlenecks()
    print(critical[['Rank', 'State', 'OverallRiskScore', 'RiskClassification']].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("BOTTLENECK SUMMARY")
    print("=" * 60)
    summary = identifier.generate_bottleneck_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
