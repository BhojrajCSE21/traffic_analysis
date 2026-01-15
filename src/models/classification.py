"""
Congestion Classification Module
Classify traffic conditions based on accident patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class CongestionClassifier:
    """Classify traffic congestion levels based on accident data"""
    
    CONGESTION_LEVELS = {
        'LOW': 0,
        'MODERATE': 1,
        'HIGH': 2,
        'SEVERE': 3
    }
    
    def __init__(self):
        self.time_data: Optional[pd.DataFrame] = None
        self.severity_data: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.label_encoder = LabelEncoder()
    
    def load_processed_data(self,
                            time_file: str = "data/processed/accidents_time_processed.csv",
                            severity_file: str = "data/processed/accidents_severity_processed.csv"):
        """Load processed datasets"""
        self.time_data = pd.read_csv(time_file)
        self.severity_data = pd.read_csv(severity_file)
        return self
    
    def create_classification_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a dataset for congestion classification
        Features: time slot characteristics, state info
        Target: congestion level (LOW, MODERATE, HIGH, SEVERE)
        """
        df = self.time_data.copy()
        
        # Calculate percentile thresholds for classification
        p25 = df['Accidents'].quantile(0.25)
        p50 = df['Accidents'].quantile(0.50)
        p75 = df['Accidents'].quantile(0.75)
        
        def classify_congestion(accidents):
            if accidents < p25:
                return 'LOW'
            elif accidents < p50:
                return 'MODERATE'
            elif accidents < p75:
                return 'HIGH'
            else:
                return 'SEVERE'
        
        df['CongestionLevel'] = df['Accidents'].apply(classify_congestion)
        
        # Create features
        features = pd.DataFrame()
        
        # Time slot encoding
        time_slots = ['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                      '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00']
        features['TimeSlotNum'] = df['TimeSlot'].apply(lambda x: time_slots.index(x) if x in time_slots else 0)
        
        # Cyclical encoding
        features['TimeSin'] = np.sin(2 * np.pi * features['TimeSlotNum'] / 8)
        features['TimeCos'] = np.cos(2 * np.pi * features['TimeSlotNum'] / 8)
        
        # Is peak hour (18:00-21:00)
        features['IsPeakHour'] = (df['TimeSlot'] == '18:00-21:00').astype(int)
        
        # Is night time
        night_slots = ['00:00-03:00', '03:00-06:00', '21:00-24:00']
        features['IsNight'] = df['TimeSlot'].isin(night_slots).astype(int)
        
        # Is rush hour (morning or evening)
        rush_slots = ['06:00-09:00', '15:00-18:00', '18:00-21:00']
        features['IsRushHour'] = df['TimeSlot'].isin(rush_slots).astype(int)
        
        # State encoding (use accident volume as a proxy)
        state_accidents = df.groupby('State')['Accidents'].sum().to_dict()
        features['StateAccidentVolume'] = df['State'].map(state_accidents)
        
        # Normalize state accident volume
        features['StateAccidentVolume'] = (features['StateAccidentVolume'] - features['StateAccidentVolume'].min()) / \
                                          (features['StateAccidentVolume'].max() - features['StateAccidentVolume'].min())
        
        target = df['CongestionLevel']
        
        return features, target
    
    def train_random_forest(self) -> Dict:
        """
        Train a Random Forest classifier
        """
        X, y = self.create_classification_dataset()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y_encoded, cv=5)
        
        metrics = {
            'model': 'Random Forest Classifier',
            'accuracy': round(accuracy, 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                             target_names=self.label_encoder.classes_,
                                             output_dict=True)
        
        self.models['random_forest'] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def train_gradient_boosting(self) -> Dict:
        """
        Train a Gradient Boosting classifier
        """
        X, y = self.create_classification_dataset()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
        
        # Train model
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y_encoded, cv=5)
        
        metrics = {
            'model': 'Gradient Boosting Classifier',
            'accuracy': round(accuracy, 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4)
        }
        
        self.models['gradient_boosting'] = {
            'model': model,
            'metrics': metrics
        }
        
        return metrics
    
    def predict_congestion(self, time_slot: str, state: str, model_name: str = 'random_forest') -> str:
        """
        Predict congestion level for a given time slot and state
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained. Train it first.")
        
        time_slots = ['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                      '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00']
        
        time_slot_num = time_slots.index(time_slot) if time_slot in time_slots else 0
        
        # Get state accident volume
        state_accidents = self.time_data.groupby('State')['Accidents'].sum()
        state_volume = state_accidents.get(state, state_accidents.mean())
        state_volume_norm = (state_volume - state_accidents.min()) / (state_accidents.max() - state_accidents.min())
        
        # Create features
        features = pd.DataFrame({
            'TimeSlotNum': [time_slot_num],
            'TimeSin': [np.sin(2 * np.pi * time_slot_num / 8)],
            'TimeCos': [np.cos(2 * np.pi * time_slot_num / 8)],
            'IsPeakHour': [1 if time_slot == '18:00-21:00' else 0],
            'IsNight': [1 if time_slot in ['00:00-03:00', '03:00-06:00', '21:00-24:00'] else 0],
            'IsRushHour': [1 if time_slot in ['06:00-09:00', '15:00-18:00', '18:00-21:00'] else 0],
            'StateAccidentVolume': [state_volume_norm]
        })
        
        model = self.models[model_name]['model']
        prediction = model.predict(features)[0]
        
        return self.label_encoder.inverse_transform([prediction])[0]
    
    def get_congestion_distribution(self) -> pd.DataFrame:
        """
        Get distribution of congestion levels across the dataset
        """
        _, y = self.create_classification_dataset()
        distribution = y.value_counts().reset_index()
        distribution.columns = ['CongestionLevel', 'Count']
        distribution['Percentage'] = (distribution['Count'] / distribution['Count'].sum() * 100).round(2)
        return distribution
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained classifiers
        """
        comparisons = []
        for name, data in self.models.items():
            metrics = data['metrics']
            comparisons.append({
                'Model': metrics['model'],
                'Accuracy': metrics['accuracy'],
                'CV_Mean': metrics['cv_mean'],
                'CV_Std': metrics['cv_std']
            })
        
        return pd.DataFrame(comparisons).sort_values('Accuracy', ascending=False)


# Quick test when run directly
if __name__ == "__main__":
    classifier = CongestionClassifier()
    classifier.load_processed_data()
    
    print("=" * 60)
    print("CONGESTION LEVEL DISTRIBUTION")
    print("=" * 60)
    print(classifier.get_congestion_distribution().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFIERS")
    print("=" * 60)
    
    rf_metrics = classifier.train_random_forest()
    print(f"Random Forest: Accuracy={rf_metrics['accuracy']}, CV={rf_metrics['cv_mean']}±{rf_metrics['cv_std']}")
    
    gb_metrics = classifier.train_gradient_boosting()
    print(f"Gradient Boosting: Accuracy={gb_metrics['accuracy']}, CV={gb_metrics['cv_mean']}±{gb_metrics['cv_std']}")
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(classifier.compare_models().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    test_cases = [
        ('18:00-21:00', 'Maharashtra'),
        ('03:00-06:00', 'Maharashtra'),
        ('09:00-12:00', 'Lakshadweep'),
    ]
    for time_slot, state in test_cases:
        prediction = classifier.predict_congestion(time_slot, state)
        print(f"{time_slot}, {state}: {prediction}")
