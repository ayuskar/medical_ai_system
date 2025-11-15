import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from django.conf import settings
from datetime import datetime

class NoShowPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model_path = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models', 'no_show_model.pkl')
        self.encoder_path = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models', 'no_show_encoders.pkl')
        self.scaler_path = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models', 'no_show_scaler.pkl')
        
    def generate_realistic_data(self):
        """Generate realistic training data based on medical no-show research"""
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'age': np.random.normal(45, 15, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.45, 0.55]),
            'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                                          n_samples, p=[0.18, 0.18, 0.18, 0.18, 0.18, 0.10]),
            'lead_time': np.random.exponential(7, n_samples).astype(int) + 1,
            'previous_no_shows': np.random.poisson(0.5, n_samples),
            'sms_reminder': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'medical_specialty': np.random.choice([
                'Cardiology', 'Neurology', 'Pediatrics', 'Orthopedics', 'General', 
                'Dermatology', 'Psychiatry', 'Dentistry'
            ], n_samples),
            'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening'], n_samples, p=[0.4, 0.5, 0.1]),
            'insurance_type': np.random.choice(['Private', 'Public', 'Self-pay'], n_samples, p=[0.6, 0.35, 0.05]),
            'distance_to_clinic': np.random.exponential(10, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate no-show probability based on realistic factors
        no_show_prob = (
            (df['age'] < 30) * 0.3 +                    # Younger patients more likely to no-show
            (df['age'] > 65) * 0.1 +                    # Older patients less likely
            (df['gender'] == 'Male') * 0.1 +            # Males slightly more likely
            (df['day_of_week'] == 'Saturday') * 0.2 +   # Weekend appointments higher risk
            (df['lead_time'] > 14) * 0.3 +              # Longer lead times higher risk
            (df['previous_no_shows'] > 0) * 0.4 +       # History of no-shows
            (df['sms_reminder'] == 0) * 0.2 +           # No SMS reminder
            (df['time_of_day'] == 'Evening') * 0.1 +    # Evening appointments
            (df['insurance_type'] == 'Self-pay') * 0.2 + # Self-pay patients
            (df['distance_to_clinic'] > 20) * 0.15      # Long distance
        ) / 3.0  # Normalize
        
        # Add some randomness
        noise = np.random.normal(0, 0.1, n_samples)
        no_show_prob = np.clip(no_show_prob + noise, 0, 1)
        
        # Generate binary no-show outcomes
        df['no_show'] = (no_show_prob > np.random.uniform(0, 1, n_samples)).astype(int)
        
        return df
    
    def train_model(self):
        """Train the enhanced no-show prediction model"""
        print("Generating realistic training data...")
        df = self.generate_realistic_data()
        
        # Prepare features and target
        categorical_columns = ['gender', 'day_of_week', 'medical_specialty', 'time_of_day', 'insurance_type']
        numerical_columns = ['age', 'lead_time', 'previous_no_shows', 'distance_to_clinic']
        
        # Encode categorical variables
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Scale numerical features
        X_numerical = self.scaler.fit_transform(df[numerical_columns])
        X_categorical = df[categorical_columns].values
        X = np.column_stack([X_numerical, X_categorical])
        
        y = df['no_show']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model with balanced class weights
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Save model and encoders
        model_dir = os.path.dirname(self.model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.label_encoders, self.encoder_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"No-show model trained with accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def load_model(self):
        """Load trained model and encoders"""
        try:
            self.model = joblib.load(self.model_path)
            self.label_encoders = joblib.load(self.encoder_path)
            self.scaler = joblib.load(self.scaler_path)
            print("No-show prediction model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading no-show model: {e}")
            return False
    
    def predict(self, features):
        """Predict no-show probability with enhanced features"""
        if not self.model:
            if not self.load_model():
                print("Training new no-show model...")
                self.train_model()
        
        # Prepare input features with all expected columns
        expected_features = [
            'age', 'gender', 'day_of_week', 'lead_time', 'previous_no_shows',
            'sms_reminder', 'medical_specialty', 'time_of_day', 'insurance_type', 'distance_to_clinic'
        ]
        
        # Set default values for missing features
        default_values = {
            'time_of_day': 'Afternoon',
            'insurance_type': 'Private',
            'distance_to_clinic': 10.0,
            'sms_reminder': 1
        }
        
        for feature in expected_features:
            if feature not in features:
                features[feature] = default_values.get(feature, 0)
        
        # Create input DataFrame
        input_df = pd.DataFrame([features])
        
        # Encode categorical features
        categorical_columns = ['gender', 'day_of_week', 'medical_specialty', 'time_of_day', 'insurance_type']
        numerical_columns = ['age', 'lead_time', 'previous_no_shows', 'distance_to_clinic']
        
        for col in categorical_columns:
            if col in input_df.columns and col in self.label_encoders:
                try:
                    input_df[col] = self.label_encoders[col].transform([features[col]])[0]
                except ValueError:
                    # Handle unseen categories
                    input_df[col] = 0
        
        # Scale numerical features
        X_numerical = self.scaler.transform(input_df[numerical_columns])
        X_categorical = input_df[categorical_columns].values
        X_processed = np.column_stack([X_numerical, X_categorical])
        
        # Make prediction
        probability = self.model.predict_proba(X_processed)[0][1]
        return round(probability, 3)

# Singleton instance
no_show_predictor = NoShowPredictor()