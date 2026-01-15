"""
Anomaly Detection Module
Detect unusual traffic patterns and outliers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Detect anomalies in traffic accident data"""
    
    def __init__(self):
        self.time_data: Optional[pd.DataFrame] = None
        self.month_data: Optional[pd.DataFrame] = None
        self.severity_data: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        self.models: Dict = {}
    
    def load_processed_data(self,
                            time_file: str = "data/processed/accidents_time_processed.csv",
                            month_file: str = "data/processed/accidents_month_processed.csv",
                            severity_file: str = "data/processed/accidents_severity_processed.csv"):
        """Load processed datasets"""
        self.time_data = pd.read_csv(time_file)
        self.month_data = pd.read_csv(month_file)
        self.severity_data = pd.read_csv(severity_file)
        return self
    
    def detect_statistical_outliers(self, data: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using statistical methods
        Methods: 'zscore', 'iqr'
        """
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            return z_scores > threshold
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_time_anomalies(self) -> pd.DataFrame:
        """
        Detect states with unusual time-based accident patterns
        """
        df = self.time_data.copy()
        
        # Calculate expected vs actual for each state-timeslot
        # Expected = mean accidents for that timeslot across all states
        expected = df.groupby('TimeSlot')['Accidents'].mean().to_dict()
        df['Expected'] = df['TimeSlot'].map(expected)
        df['Deviation'] = ((df['Accidents'] - df['Expected']) / df['Expected'] * 100).round(2)
        
        # Flag anomalies (deviation > 100% or < -50%)
        df['IsAnomaly'] = (df['Deviation'] > 100) | (df['Deviation'] < -50)
        
        # Get anomalies
        anomalies = df[df['IsAnomaly']][['State', 'TimeSlot', 'Accidents', 'Expected', 'Deviation']]
        anomalies = anomalies.sort_values('Deviation', ascending=False)
        
        return anomalies
    
    def detect_monthly_anomalies(self) -> pd.DataFrame:
        """
        Detect states with unusual monthly accident patterns
        """
        df = self.month_data.copy()
        
        # Calculate expected vs actual for each state-month
        expected = df.groupby('Month')['Accidents'].mean().to_dict()
        df['Expected'] = df['Month'].map(expected)
        df['Deviation'] = ((df['Accidents'] - df['Expected']) / df['Expected'] * 100).round(2)
        
        # Flag anomalies (deviation > 100% or < -50%)
        df['IsAnomaly'] = (df['Deviation'] > 100) | (df['Deviation'] < -50)
        
        # Get anomalies
        anomalies = df[df['IsAnomaly']][['State', 'Month', 'Accidents', 'Expected', 'Deviation']]
        anomalies = anomalies.sort_values('Deviation', ascending=False)
        
        return anomalies
    
    def detect_severity_anomalies(self) -> pd.DataFrame:
        """
        Detect states with unusual fatality/injury rates
        """
        df = self.severity_data.copy()
        
        # Calculate z-scores for fatality and injury rates
        df['FatalityZScore'] = stats.zscore(df['FatalityRate'])
        df['InjuryZScore'] = stats.zscore(df['InjuryRate'])
        
        # Flag anomalies
        df['FatalityAnomaly'] = np.abs(df['FatalityZScore']) > 2
        df['InjuryAnomaly'] = np.abs(df['InjuryZScore']) > 2
        df['IsAnomaly'] = df['FatalityAnomaly'] | df['InjuryAnomaly']
        
        anomalies = df[df['IsAnomaly']][['State', 'Cases', 'FatalityRate', 'InjuryRate', 
                                          'FatalityZScore', 'InjuryZScore']]
        anomalies = anomalies.sort_values('FatalityZScore', ascending=False)
        
        return anomalies
    
    def train_isolation_forest(self) -> Dict:
        """
        Train Isolation Forest model for anomaly detection
        """
        df = self.time_data.copy()
        
        # Create feature matrix
        pivot = df.pivot_table(index='State', columns='TimeSlot', values='Accidents', aggfunc='sum').fillna(0)
        
        # Normalize
        X_scaled = self.scaler.fit_transform(pivot)
        
        # Train Isolation Forest
        model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        predictions = model.fit_predict(X_scaled)
        
        # Add results to dataframe
        result = pivot.copy()
        result['AnomalyScore'] = model.decision_function(X_scaled)
        result['IsAnomaly'] = predictions == -1
        result['State'] = pivot.index
        
        # Get anomalies
        anomalies = result[result['IsAnomaly']][['State', 'AnomalyScore']].reset_index(drop=True)
        anomalies = anomalies.sort_values('AnomalyScore')
        
        self.models['isolation_forest'] = {
            'model': model,
            'scaler': self.scaler,
            'anomalies': anomalies
        }
        
        return {
            'total_states': len(pivot),
            'anomalies_detected': len(anomalies),
            'contamination': 0.1,
            'anomaly_states': anomalies['State'].tolist()
        }
    
    def get_state_risk_profile(self, state: str) -> Dict:
        """
        Generate risk profile for a specific state
        """
        # Time-based analysis
        time_df = self.time_data[self.time_data['State'] == state]
        national_avg = self.time_data.groupby('TimeSlot')['Accidents'].mean()
        
        time_comparison = time_df.set_index('TimeSlot')['Accidents'] / national_avg
        above_avg_slots = time_comparison[time_comparison > 1.5].index.tolist()
        
        # Severity analysis
        severity_df = self.severity_data[self.severity_data['State'] == state]
        if len(severity_df) > 0:
            row = severity_df.iloc[0]
            fatality_rate = row['FatalityRate']
            national_fatality = self.severity_data['FatalityRate'].mean()
        else:
            fatality_rate = 0
            national_fatality = 0
        
        # Monthly analysis
        month_df = self.month_data[self.month_data['State'] == state]
        if len(month_df) > 0:
            peak_month = month_df.loc[month_df['Accidents'].idxmax(), 'Month']
            trough_month = month_df.loc[month_df['Accidents'].idxmin(), 'Month']
        else:
            peak_month = 'N/A'
            trough_month = 'N/A'
        
        return {
            'state': state,
            'high_risk_time_slots': above_avg_slots,
            'fatality_rate': fatality_rate,
            'fatality_vs_national': round(fatality_rate / national_fatality, 2) if national_fatality > 0 else 0,
            'peak_accident_month': peak_month,
            'lowest_accident_month': trough_month,
            'total_accidents': time_df['Accidents'].sum()
        }
    
    def generate_anomaly_summary(self) -> Dict:
        """
        Generate comprehensive anomaly summary
        """
        time_anomalies = self.detect_time_anomalies()
        monthly_anomalies = self.detect_monthly_anomalies()
        severity_anomalies = self.detect_severity_anomalies()
        iso_result = self.train_isolation_forest()
        
        return {
            'time_anomalies_count': len(time_anomalies),
            'monthly_anomalies_count': len(monthly_anomalies),
            'severity_anomalies_count': len(severity_anomalies),
            'isolation_forest_anomalies': iso_result['anomaly_states'],
            'most_deviant_time_patterns': time_anomalies.head(5)['State'].tolist() if len(time_anomalies) > 0 else [],
            'highest_fatality_anomalies': severity_anomalies.head(3)['State'].tolist() if len(severity_anomalies) > 0 else []
        }


# Quick test when run directly
if __name__ == "__main__":
    detector = AnomalyDetector()
    detector.load_processed_data()
    
    print("=" * 60)
    print("TIME-BASED ANOMALIES (Top 10)")
    print("=" * 60)
    time_anomalies = detector.detect_time_anomalies()
    print(time_anomalies.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SEVERITY ANOMALIES")
    print("=" * 60)
    severity_anomalies = detector.detect_severity_anomalies()
    print(severity_anomalies.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("ISOLATION FOREST RESULTS")
    print("=" * 60)
    iso_result = detector.train_isolation_forest()
    print(f"Anomalies detected: {iso_result['anomalies_detected']} out of {iso_result['total_states']} states")
    print(f"Anomaly states: {iso_result['anomaly_states']}")
    
    print("\n" + "=" * 60)
    print("SAMPLE STATE RISK PROFILE")
    print("=" * 60)
    profile = detector.get_state_risk_profile('Maharashtra')
    for key, value in profile.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("ANOMALY SUMMARY")
    print("=" * 60)
    summary = detector.generate_anomaly_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
