"""
Enhanced No-Show Prediction Model with Multiple ML Algorithms
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
import os
from django.conf import settings

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class EnhancedNoShowPredictor:
    """
    Advanced no-show prediction with ensemble learning and feature engineering
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_loaded = False
        
        # Model paths
        self.model_dir = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models')
        self.model_path = os.path.join(self.model_dir, 'enhanced_no_show_model.pkl')
        self.preprocessor_path = os.path.join(self.model_dir, 'no_show_preprocessor.pkl')
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model
        self.load_model()
    
    def generate_enhanced_training_data(self, n_samples: int = 10000):
        """
        Generate realistic training data with more features and patterns
        """
        print(f"Generating enhanced training data with {n_samples} samples...")
        np.random.seed(42)
        
        # Base features
        data = {
            'age': np.random.normal(45, 15, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.50, 0.02]),
            'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                                          n_samples, p=[0.18, 0.18, 0.18, 0.18, 0.18, 0.10]),
            'lead_time': np.random.exponential(10, n_samples).astype(int) + 1,
            'previous_no_shows': np.random.poisson(0.7, n_samples),
            'sms_reminder': np.random.choice([0, 1], n_samples, p=[0.25, 0.75]),
            'email_reminder': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'medical_specialty': np.random.choice([
                'Cardiology', 'Neurology', 'Pediatrics', 'Orthopedics', 'General Practice',
                'Dermatology', 'Psychiatry', 'Dentistry', 'Ophthalmology', 'ENT'
            ], n_samples),
            'time_of_day': np.random.choice(['Morning (8-12)', 'Afternoon (12-4)', 'Evening (4-8)'], 
                                          n_samples, p=[0.4, 0.5, 0.1]),
            'insurance_type': np.random.choice(['Private', 'Public', 'Self-pay', 'Corporate'], 
                                             n_samples, p=[0.55, 0.35, 0.05, 0.05]),
            'distance_to_clinic': np.random.exponential(15, n_samples),
            'appointment_type': np.random.choice(['New Patient', 'Follow-up', 'Consultation', 'Procedure'], 
                                               n_samples, p=[0.2, 0.5, 0.2, 0.1]),
            'weather_condition': np.random.choice(['Clear', 'Rainy', 'Snowy', 'Stormy'], 
                                                n_samples, p=[0.7, 0.15, 0.1, 0.05]),
            'transportation_mode': np.random.choice(['Car', 'Public Transport', 'Walk', 'Taxi'], 
                                                  n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            'patient_occupation': np.random.choice(['Employed', 'Self-employed', 'Student', 'Retired', 'Unemployed'], 
                                                 n_samples, p=[0.5, 0.15, 0.15, 0.15, 0.05]),
            'season': np.random.choice(['Spring', 'Summer', 'Autumn', 'Winter'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['is_weekend'] = df['day_of_day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], 
                                labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
        df['lead_time_group'] = pd.cut(df['lead_time'], bins=[0, 2, 7, 14, 30, 365],
                                      labels=['Same Day', '1-2 Days', '3-7 Days', '1-2 Weeks', '1 Month+'])
        df['distance_group'] = pd.cut(df['distance_to_clinic'], bins=[0, 5, 10, 20, 50, 1000],
                                     labels=['<5km', '5-10km', '10-20km', '20-50km', '>50km'])
        
        # Add interaction features
        df['has_reminder'] = df['sms_reminder'] | df['email_reminder']
        df['high_risk_specialty'] = df['medical_specialty'].apply(
            lambda x: 1 if x in ['Psychiatry', 'Dentistry', 'General Practice'] else 0
        )
        df['bad_weather'] = df['weather_condition'].apply(lambda x: 1 if x in ['Rainy', 'Snowy', 'Stormy'] else 0)
        
        # Calculate no-show probability with complex rules
        no_show_prob = (
            (df['age'] < 25) * 0.25 +                    # Young patients
            (df['age'] > 65) * 0.15 +                    # Elderly patients
            (df['gender'] == 'Male') * 0.1 +             # Gender factor
            (df['is_weekend'] == 1) * 0.2 +              # Weekend appointments
            (df['lead_time'] > 21) * 0.3 +               # Long lead time
            (df['lead_time'] < 2) * 0.1 +                # Very short notice
            (df['previous_no_shows'] == 0) * -0.2 +      # Good history
            (df['previous_no_shows'] > 2) * 0.5 +        # Bad history
            (df['has_reminder'] == 0) * 0.25 +           # No reminders
            (df['high_risk_specialty'] == 1) * 0.15 +    # High-risk specialty
            (df['time_of_day'] == 'Evening (4-8)') * 0.1 + # Evening appointments
            (df['insurance_type'] == 'Self-pay') * 0.2 +  # Self-pay patients
            (df['distance_to_clinic'] > 25) * 0.15 +      # Long distance
            (df['appointment_type'] == 'New Patient') * 0.1 + # New patients
            (df['bad_weather'] == 1) * 0.2 +              # Bad weather
            (df['transportation_mode'] == 'Public Transport') * 0.1 + # Public transport
            (df['patient_occupation'] == 'Unemployed') * 0.15 + # Unemployed
            (np.random.normal(0, 0.1, n_samples))         # Random noise
        ) / 5.0  # Normalize
        
        # Apply sigmoid to get probabilities between 0 and 1
        no_show_prob = 1 / (1 + np.exp(-no_show_prob))
        
        # Generate binary outcomes
        df['no_show'] = (no_show_prob > np.random.uniform(0, 1, n_samples)).astype(int)
        
        print(f"Generated dataset with {df['no_show'].sum()} no-shows ({df['no_show'].mean():.1%})")
        return df
    
    def create_feature_pipeline(self, df: pd.DataFrame):
        """Create preprocessing pipeline for features"""
        # Separate features and target
        X = df.drop('no_show', axis=1)
        y = df['no_show']
        
        # Identify column types
        numerical_features = ['age', 'lead_time', 'previous_no_shows', 'distance_to_clinic', 
                            'sms_reminder', 'email_reminder', 'has_reminder', 'high_risk_specialty',
                            'bad_weather', 'is_weekend']
        
        categorical_features = ['gender', 'day_of_week', 'medical_specialty', 'time_of_day',
                              'insurance_type', 'appointment_type', 'weather_condition',
                              'transportation_mode', 'patient_occupation', 'season',
                              'age_group', 'lead_time_group', 'distance_group']
        
        # Create transformers
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor, X, y
    
    def train_ensemble_model(self):
        """Train ensemble model with multiple algorithms"""
        print("Training enhanced no-show prediction model...")
        
        # Generate training data
        df = self.generate_enhanced_training_data(n_samples=15000)
        
        # Create preprocessing pipeline
        preprocessor, X, y = self.create_feature_pipeline(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create individual models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }
        
        # Create pipeline with preprocessor and model
        pipelines = {}
        for name, model in models.items():
            pipelines[name] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
        
        # Train individual models and evaluate
        print("\nTraining individual models...")
        model_performance = {}
        
        for name, pipeline in pipelines.items():
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            model_performance[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
            
            print(f"  Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {roc_auc:.3f}")
        
        # Create ensemble model (voting classifier)
        print("\nCreating ensemble model...")
        voting_classifier = VotingClassifier(
            estimators=[(name, pipelines[name]) for name in models.keys()],
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble model
        voting_classifier.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = voting_classifier.predict(X_test)
        y_pred_proba_ensemble = voting_classifier.predict_proba(X_test)[:, 1]
        
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_f1 = f1_score(y_test, y_pred_ensemble)
        ensemble_auc = roc_auc_score(y_test, y_pred_proba_ensemble)
        
        print(f"\nEnsemble Model Performance:")
        print(f"  Accuracy: {ensemble_accuracy:.3f}")
        print(f"  F1 Score: {ensemble_f1:.3f}")
        print(f"  ROC AUC: {ensemble_auc:.3f}")
        
        # Save the ensemble model
        self.model = voting_classifier
        self.preprocessor = preprocessor
        self.feature_names = list(X.columns)
        
        # Save model components
        self.save_model()
        
        return {
            'ensemble': {
                'accuracy': ensemble_accuracy,
                'f1': ensemble_f1,
                'roc_auc': ensemble_auc
            },
            'individual_models': model_performance
        }
    
    def save_model(self):
        """Save trained model and preprocessor"""
        try:
            # Save the entire pipeline
            joblib.dump({
                'model': self.model,
                'preprocessor': self.preprocessor,
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat()
            }, self.model_path)
            
            print(f"Model saved to {self.model_path}")
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                print("Loading trained no-show prediction model...")
                saved_data = joblib.load(self.model_path)
                
                self.model = saved_data['model']
                self.preprocessor = saved_data['preprocessor']
                self.feature_names = saved_data['feature_names']
                self.model_loaded = True
                
                print("✓ Model loaded successfully!")
                return True
            else:
                print("No trained model found. Training new model...")
                self.train_ensemble_model()
                return True
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, patient_data: Dict) -> Dict:
        """
        Predict no-show probability for a given patient
        Returns comprehensive prediction with explanations
        """
        if not self.model_loaded:
            if not self.load_model():
                return {
                    'prediction': 0.5,
                    'confidence': 0.5,
                    'risk_level': 'medium',
                    'factors': [],
                    'error': 'Model not loaded'
                }
        
        try:
            # Create DataFrame with all expected features
            default_values = {
                'age': 45,
                'gender': 'Male',
                'day_of_week': 'Monday',
                'lead_time': 7,
                'previous_no_shows': 0,
                'sms_reminder': 1,
                'email_reminder': 0,
                'medical_specialty': 'General Practice',
                'time_of_day': 'Afternoon (12-4)',
                'insurance_type': 'Private',
                'distance_to_clinic': 10.0,
                'appointment_type': 'Follow-up',
                'weather_condition': 'Clear',
                'transportation_mode': 'Car',
                'patient_occupation': 'Employed',
                'season': 'Spring',
                'is_weekend': 0,
                'age_group': 'Adult',
                'lead_time_group': '3-7 Days',
                'distance_group': '5-10km',
                'has_reminder': 1,
                'high_risk_specialty': 0,
                'bad_weather': 0
            }
            
            # Update with provided data
            input_data = default_values.copy()
            input_data.update(patient_data)
            
            # Calculate derived features
            input_data['is_weekend'] = 1 if input_data['day_of_week'] in ['Saturday', 'Sunday'] else 0
            input_data['has_reminder'] = input_data['sms_reminder'] | input_data['email_reminder']
            input_data['high_risk_specialty'] = 1 if input_data['medical_specialty'] in [
                'Psychiatry', 'Dentistry', 'General Practice'
            ] else 0
            input_data['bad_weather'] = 1 if input_data['weather_condition'] in [
                'Rainy', 'Snowy', 'Stormy'
            ] else 0
            
            # Age group
            age = input_data['age']
            if age < 18:
                input_data['age_group'] = 'Child'
            elif age < 30:
                input_data['age_group'] = 'Young Adult'
            elif age < 45:
                input_data['age_group'] = 'Adult'
            elif age < 60:
                input_data['age_group'] = 'Senior'
            else:
                input_data['age_group'] = 'Elderly'
            
            # Lead time group
            lead_time = input_data['lead_time']
            if lead_time <= 2:
                input_data['lead_time_group'] = 'Same Day'
            elif lead_time <= 7:
                input_data['lead_time_group'] = '1-2 Days'
            elif lead_time <= 14:
                input_data['lead_time_group'] = '3-7 Days'
            elif lead_time <= 30:
                input_data['lead_time_group'] = '1-2 Weeks'
            else:
                input_data['lead_time_group'] = '1 Month+'
            
            # Distance group
            distance = input_data['distance_to_clinic']
            if distance <= 5:
                input_data['distance_group'] = '<5km'
            elif distance <= 10:
                input_data['distance_group'] = '5-10km'
            elif distance <= 20:
                input_data['distance_group'] = '10-20km'
            elif distance <= 50:
                input_data['distance_group'] = '20-50km'
            else:
                input_data['distance_group'] = '>50km'
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = default_values.get(feature, 0)
            
            # Reorder columns to match training
            input_df = input_df[self.feature_names]
            
            # Make prediction
            probability = self.model.predict_proba(input_df)[0][1]
            
            # Get feature importance for explanation
            factors = self.explain_prediction(input_data, probability)
            
            # Determine risk level
            if probability > 0.7:
                risk_level = 'high'
                recommendation = 'Consider calling patient to confirm appointment'
            elif probability > 0.4:
                risk_level = 'medium'
                recommendation = 'Send reminder SMS/email 24 hours before'
            else:
                risk_level = 'low'
                recommendation = 'Standard reminder procedure'
            
            return {
                'prediction': round(probability, 3),
                'confidence': round(min(probability, 1 - probability) * 2, 3),
                'risk_level': risk_level,
                'recommendation': recommendation,
                'factors': factors,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'prediction': 0.5,
                'confidence': 0.5,
                'risk_level': 'medium',
                'factors': [],
                'error': str(e)
            }
    
    def explain_prediction(self, patient_data: Dict, probability: float) -> List[Dict]:
        """Explain the prediction by identifying key contributing factors"""
        factors = []
        
        # Analyze each feature's contribution
        if patient_data['previous_no_shows'] > 0:
            factors.append({
                'feature': 'Previous No-Shows',
                'value': patient_data['previous_no_shows'],
                'impact': 'high',
                'description': f'Patient has {patient_data["previous_no_shows"]} previous no-show(s)'
            })
        
        if patient_data['lead_time'] > 21:
            factors.append({
                'feature': 'Long Lead Time',
                'value': f'{patient_data["lead_time"]} days',
                'impact': 'high',
                'description': 'Appointment scheduled far in advance increases risk'
            })
        
        if patient_data['has_reminder'] == 0:
            factors.append({
                'feature': 'No Reminders',
                'value': 'No SMS/email reminders',
                'impact': 'medium',
                'description': 'Patient will not receive appointment reminders'
            })
        
        if patient_data['high_risk_specialty'] == 1:
            factors.append({
                'feature': 'Specialty Risk',
                'value': patient_data['medical_specialty'],
                'impact': 'medium',
                'description': 'This specialty has higher no-show rates'
            })
        
        if patient_data['distance_to_clinic'] > 25:
            factors.append({
                'feature': 'Distance',
                'value': f'{patient_data["distance_to_clinic"]} km',
                'impact': 'medium',
                'description': 'Long travel distance increases no-show risk'
            })
        
        if patient_data['insurance_type'] == 'Self-pay':
            factors.append({
                'feature': 'Insurance',
                'value': 'Self-pay',
                'impact': 'medium',
                'description': 'Self-paying patients have higher no-show rates'
            })
        
        # Add probability-based factor
        if probability > 0.7:
            factors.append({
                'feature': 'Overall Risk',
                'value': 'High',
                'impact': 'high',
                'description': 'Multiple risk factors indicate high no-show probability'
            })
        
        return factors

# Singleton instance
no_show_predictor = EnhancedNoShowPredictor()