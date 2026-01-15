"""
Traffic Forecasting Module
Time-series forecasting for traffic accident prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TrafficForecaster:
    """Forecast traffic accidents using various models"""
    
    def __init__(self):
        self.month_data: Optional[pd.DataFrame] = None
        self.time_data: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.scaler = StandardScaler()
    
    def load_processed_data(self,
                            month_file: str = "data/processed/accidents_month_processed.csv",
                            time_file: str = "data/processed/accidents_time_processed.csv"):
        """Load processed datasets"""
        self.month_data = pd.read_csv(month_file)
        self.time_data = pd.read_csv(time_file)
        return self
    
    def prepare_features(self, state: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for forecasting from monthly data
        """
        df = self.month_data.copy()
        
        if state:
            df = df[df['State'] == state]
        
        # Aggregate by month
        monthly = df.groupby(['Month', 'MonthNum']).agg({
            'Accidents': 'sum'
        }).reset_index().sort_values('MonthNum')
        
        # Create features
        features = pd.DataFrame()
        features['MonthNum'] = monthly['MonthNum']
        
        # Cyclical encoding for months
        features['MonthSin'] = np.sin(2 * np.pi * monthly['MonthNum'] / 12)
        features['MonthCos'] = np.cos(2 * np.pi * monthly['MonthNum'] / 12)
        
        # Quarter indicator
        features['Quarter'] = ((monthly['MonthNum'] - 1) // 3) + 1
        
        # Is peak month (based on historical data)
        peak_months = [5, 10, 11]  # May, October, November typically higher
        features['IsPeakMonth'] = monthly['MonthNum'].isin(peak_months).astype(int)
        
        # Lag features (previous month accidents)
        monthly['Lag1'] = monthly['Accidents'].shift(1).fillna(monthly['Accidents'].mean())
        monthly['Lag2'] = monthly['Accidents'].shift(2).fillna(monthly['Accidents'].mean())
        features['Lag1'] = monthly['Lag1']
        features['Lag2'] = monthly['Lag2']
        
        # Rolling mean
        features['RollingMean3'] = monthly['Accidents'].rolling(window=3, min_periods=1).mean()
        
        target = monthly['Accidents']
        
        return features, target
    
    def train_linear_regression(self, state: Optional[str] = None) -> Dict:
        """
        Train a Linear Regression model
        """
        X, y = self.prepare_features(state)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        metrics = {
            'model': 'Linear Regression',
            'mae': round(mean_absolute_error(y_test, y_pred), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'r2': round(r2_score(y_test, y_pred), 4)
        }
        
        self.models['linear_regression'] = {
            'model': model,
            'scaler': self.scaler,
            'metrics': metrics
        }
        
        return metrics
    
    def train_random_forest(self, state: Optional[str] = None) -> Dict:
        """
        Train a Random Forest model
        """
        X, y = self.prepare_features(state)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Train model (no scaling needed for RF)
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'model': 'Random Forest',
            'mae': round(mean_absolute_error(y_test, y_pred), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'r2': round(r2_score(y_test, y_pred), 4)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.models['random_forest'] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        return metrics
    
    def forecast_next_months(self, n_months: int = 3, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Forecast accidents for the next n months
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained. Train it first.")
        
        X, y = self.prepare_features()
        model_data = self.models[model_name]
        model = model_data['model']
        
        # Get last known values for lag features
        last_accidents = y.iloc[-1]
        second_last = y.iloc[-2] if len(y) > 1 else last_accidents
        rolling_mean = y.iloc[-3:].mean()
        
        forecasts = []
        current_month = 12  # Start from December (assuming data ends there)
        
        for i in range(n_months):
            next_month = (current_month % 12) + 1
            
            # Create features for next month
            features = pd.DataFrame({
                'MonthNum': [next_month],
                'MonthSin': [np.sin(2 * np.pi * next_month / 12)],
                'MonthCos': [np.cos(2 * np.pi * next_month / 12)],
                'Quarter': [((next_month - 1) // 3) + 1],
                'IsPeakMonth': [1 if next_month in [5, 10, 11] else 0],
                'Lag1': [last_accidents],
                'Lag2': [second_last],
                'RollingMean3': [rolling_mean]
            })
            
            # Predict
            if model_name == 'linear_regression':
                scaler = model_data['scaler']
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
            else:
                prediction = model.predict(features)[0]
            
            forecasts.append({
                'Month': next_month,
                'MonthName': ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December'][next_month - 1],
                'ForecastedAccidents': round(prediction)
            })
            
            # Update lag values for next iteration
            second_last = last_accidents
            last_accidents = prediction
            rolling_mean = (rolling_mean * 2 + prediction) / 3
            current_month = next_month
        
        return pd.DataFrame(forecasts)
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models
        """
        comparisons = []
        for name, data in self.models.items():
            metrics = data['metrics']
            comparisons.append({
                'Model': metrics['model'],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R2': metrics['r2']
            })
        
        return pd.DataFrame(comparisons).sort_values('R2', ascending=False)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from Random Forest model
        """
        if 'random_forest' in self.models:
            return self.models['random_forest']['feature_importance']
        return pd.DataFrame()


# Quick test when run directly
if __name__ == "__main__":
    forecaster = TrafficForecaster()
    forecaster.load_processed_data()
    
    print("=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    lr_metrics = forecaster.train_linear_regression()
    print(f"Linear Regression: MAE={lr_metrics['mae']}, R2={lr_metrics['r2']}")
    
    rf_metrics = forecaster.train_random_forest()
    print(f"Random Forest: MAE={rf_metrics['mae']}, R2={rf_metrics['r2']}")
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(forecaster.compare_models().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    print(forecaster.get_feature_importance().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("3-MONTH FORECAST")
    print("=" * 60)
    print(forecaster.forecast_next_months(3).to_string(index=False))
