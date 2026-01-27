"""
Advanced AI Symptom Checker for MediCare AI
Author: [Your Name]
Date: 2024
Description: Ensemble ML-based symptom analysis with comprehensive medical knowledge base
"""

import pandas as pd
import numpy as np
import re
import joblib
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from django.conf import settings

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class SymptomChecker:
    """Advanced symptom checker using ensemble machine learning"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.disease_database = self.create_comprehensive_database()
        self.is_trained = False
        self.model_path = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models')
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def create_comprehensive_database(self):
        """Create a comprehensive medical knowledge database"""
        return {
            # Respiratory Diseases
            'Common Cold': {
                'symptoms': ['runny nose', 'sneezing', 'sore throat', 'cough', 'congestion', 'mild fever', 'headache'],
                'severity': 2,
                'urgency': 'low',
                'body_system': 'respiratory'
            },
            'Influenza (Flu)': {
                'symptoms': ['fever', 'chills', 'muscle aches', 'cough', 'sore throat', 'headache', 'fatigue'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'respiratory'
            },
            'COVID-19': {
                'symptoms': ['fever', 'dry cough', 'tiredness', 'loss of taste', 'loss of smell', 'sore throat', 'headache'],
                'severity': 4,
                'urgency': 'high',
                'body_system': 'respiratory'
            },
            'Pneumonia': {
                'symptoms': ['fever', 'chills', 'cough with phlegm', 'chest pain', 'shortness of breath', 'fatigue'],
                'severity': 4,
                'urgency': 'high',
                'body_system': 'respiratory'
            },
            'Bronchitis': {
                'symptoms': ['cough', 'mucus production', 'fatigue', 'shortness of breath', 'chest discomfort'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'respiratory'
            },
            'Asthma': {
                'symptoms': ['wheezing', 'shortness of breath', 'chest tightness', 'coughing'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'respiratory'
            },
            
            # Cardiovascular Diseases
            'Heart Attack': {
                'symptoms': ['chest pain', 'shortness of breath', 'nausea', 'lightheadedness', 'pain in arms'],
                'severity': 5,
                'urgency': 'emergency',
                'body_system': 'cardiovascular'
            },
            'Hypertension': {
                'symptoms': ['headaches', 'shortness of breath', 'nosebleeds', 'dizziness', 'chest pain'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'cardiovascular'
            },
            
            # Gastrointestinal Diseases
            'Gastroenteritis': {
                'symptoms': ['diarrhea', 'vomiting', 'abdominal pain', 'nausea', 'fever'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'gastrointestinal'
            },
            'Food Poisoning': {
                'symptoms': ['nausea', 'vomiting', 'watery diarrhea', 'abdominal pain', 'fever'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'gastrointestinal'
            },
            'Irritable Bowel Syndrome': {
                'symptoms': ['abdominal pain', 'bloating', 'gas', 'diarrhea', 'constipation'],
                'severity': 2,
                'urgency': 'low',
                'body_system': 'gastrointestinal'
            },
            
            # Neurological Diseases
            'Migraine': {
                'symptoms': ['headache', 'nausea', 'sensitivity to light', 'sensitivity to sound', 'aura'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'neurological'
            },
            'Tension Headache': {
                'symptoms': ['headache', 'pressure around forehead', 'tenderness in scalp', 'neck pain'],
                'severity': 2,
                'urgency': 'low',
                'body_system': 'neurological'
            },
            
            # Musculoskeletal Diseases
            'Arthritis': {
                'symptoms': ['joint pain', 'stiffness', 'swelling', 'redness', 'decreased range of motion'],
                'severity': 2,
                'urgency': 'low',
                'body_system': 'musculoskeletal'
            },
            
            # Endocrine Diseases
            'Diabetes': {
                'symptoms': ['increased thirst', 'frequent urination', 'hunger', 'fatigue', 'blurred vision'],
                'severity': 4,
                'urgency': 'high',
                'body_system': 'endocrine'
            },
            
            # Dermatological Diseases
            'Eczema': {
                'symptoms': ['itchy skin', 'redness', 'dry skin', 'cracks', 'swelling'],
                'severity': 2,
                'urgency': 'low',
                'body_system': 'dermatological'
            },
            'Psoriasis': {
                'symptoms': ['red patches', 'silvery scales', 'dry skin', 'itching', 'thickened nails'],
                'severity': 2,
                'urgency': 'low',
                'body_system': 'dermatological'
            },
            
            # Infectious Diseases
            'Malaria': {
                'symptoms': ['fever', 'chills', 'headache', 'nausea', 'vomiting', 'muscle pain'],
                'severity': 4,
                'urgency': 'high',
                'body_system': 'infectious'
            },
            
            # Mental Health
            'Depression': {
                'symptoms': ['sadness', 'loss of interest', 'fatigue', 'sleep changes', 'appetite changes'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'mental_health'
            },
            'Anxiety Disorder': {
                'symptoms': ['worry', 'restlessness', 'fatigue', 'concentration problems', 'irritability'],
                'severity': 3,
                'urgency': 'moderate',
                'body_system': 'mental_health'
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess and clean symptom text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Expand common medical abbreviations
        medical_terms = {
            'temp': 'temperature',
            'hr': 'heart rate',
            'bp': 'blood pressure',
            'sob': 'shortness of breath',
            'c/o': 'complaining of',
            'dx': 'diagnosis'
        }
        
        words = text.split()
        processed_words = []
        for word in words:
            if word in medical_terms:
                processed_words.append(medical_terms[word])
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def generate_training_data(self, num_samples_per_disease: int = 200):
        """Generate comprehensive training data"""
        print(f"Generating training data with {num_samples_per_disease} samples per disease...")
        
        data = []
        for disease, info in self.disease_database.items():
            symptoms = info['symptoms']
            
            for _ in range(num_samples_per_disease):
                # Randomly select symptoms
                num_selected = np.random.randint(2, min(6, len(symptoms)) + 1)
                selected_symptoms = np.random.choice(symptoms, num_selected, replace=False)
                
                # Create multiple text variations
                variations = [
                    # Natural language variations
                    f"I have been experiencing {', '.join(selected_symptoms[:-1])} and {selected_symptoms[-1]}",
                    f"Symptoms include {', '.join(selected_symptoms)}",
                    f"Feeling {selected_symptoms[0]} along with {', '.join(selected_symptoms[1:])}",
                    f"Patient presents with {', '.join(selected_symptoms)}",
                    f"Complaining of {selected_symptoms[0]}, {selected_symptoms[1]}",
                    
                    # With duration
                    f"I have had {', '.join(selected_symptoms)} for the past few days",
                    f"Experiencing {', '.join(selected_symptoms)} since yesterday",
                    
                    # With severity
                    f"Severe {selected_symptoms[0]} with {', '.join(selected_symptoms[1:])}",
                    f"Mild {selected_symptoms[0]} and moderate {selected_symptoms[1]}",
                ]
                
                for text in variations[:3]:  # Take 3 variations per sample
                    # Add some noise/misspellings
                    if np.random.random() > 0.7:
                        text = text.replace('and', '&').replace('with', 'w/')
                    
                    data.append({
                        'text': text,
                        'disease': disease,
                        'symptoms': '|'.join(selected_symptoms),
                        'severity': info['severity']
                    })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} training samples for {len(self.disease_database)} diseases")
        return df
    
    def train_model(self):
        """Train the ensemble symptom classifier"""
        print("Starting model training...")
        
        # Generate training data
        df = self.generate_training_data(num_samples_per_disease=150)
        
        # Split data
        X = df['text']
        y = df['disease']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        print("Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train ensemble of models
        print("Training ensemble models...")
        
        # Model 1: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_tfidf, y_train_encoded)
        
        # Model 2: Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train_tfidf, y_train_encoded)
        
        # Model 3: Naive Bayes
        nb_model = MultinomialNB(alpha=0.1)
        nb_model.fit(X_train_tfidf, y_train_encoded)
        
        # Create ensemble by averaging probabilities
        self.model = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'naive_bayes': nb_model
        }
        
        # Evaluate models
        print("\nModel Evaluation:")
        print("=" * 50)
        
        for name, model in self.model.items():
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            print(f"{name}: Accuracy = {accuracy:.4f}")
        
        # Ensemble prediction (average probabilities)
        rf_proba = rf_model.predict_proba(X_test_tfidf)
        gb_proba = gb_model.predict_proba(X_test_tfidf)
        nb_proba = nb_model.predict_proba(X_test_tfidf)
        
        ensemble_proba = (rf_proba + gb_proba + nb_proba) / 3
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ensemble_accuracy = accuracy_score(y_test_encoded, ensemble_pred)
        
        print(f"\nEnsemble Model: Accuracy = {ensemble_accuracy:.4f}")
        print("=" * 50)
        
        # Save models
        self.save_model()
        
        self.is_trained = True
        print("Model training completed successfully!")
        
        return ensemble_accuracy
    
    def save_model(self):
        """Save trained model components"""
        try:
            # Save individual models
            for name, model in self.model.items():
                joblib.dump(model, os.path.join(self.model_path, f'{name}_model.pkl'))
            
            # Save vectorizer and encoder
            joblib.dump(self.vectorizer, os.path.join(self.model_path, 'vectorizer.pkl'))
            joblib.dump(self.label_encoder, os.path.join(self.model_path, 'label_encoder.pkl'))
            
            # Save disease database
            with open(os.path.join(self.model_path, 'disease_database.json'), 'w') as f:
                json.dump(self.disease_database, f, indent=2)
            
            print(f"Models saved to {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_model(self):
        """Load trained model components"""
        try:
            print("Attempting to load pre-trained models...")
            
            # Check if model files exist
            required_files = [
                'random_forest_model.pkl',
                'gradient_boosting_model.pkl', 
                'naive_bayes_model.pkl',
                'vectorizer.pkl',
                'label_encoder.pkl',
                'disease_database.json'
            ]
            
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(self.model_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"Missing model files: {missing_files}")
                return False
            
            # Load models
            self.model = {
                'random_forest': joblib.load(os.path.join(self.model_path, 'random_forest_model.pkl')),
                'gradient_boosting': joblib.load(os.path.join(self.model_path, 'gradient_boosting_model.pkl')),
                'naive_bayes': joblib.load(os.path.join(self.model_path, 'naive_bayes_model.pkl'))
            }
            
            # Load vectorizer and encoder
            self.vectorizer = joblib.load(os.path.join(self.model_path, 'vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_path, 'label_encoder.pkl'))
            
            # Load disease database
            with open(os.path.join(self.model_path, 'disease_database.json'), 'r') as f:
                self.disease_database = json.load(f)
            
            self.is_trained = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_ensemble(self, symptoms_text: str, top_k: int = 5):
        """Predict using ensemble of models"""
        if not self.is_trained:
            print("Model not trained. Training now...")
            self.train_model()
        
        # Preprocess text
        cleaned_text = self.preprocess_text(symptoms_text)
        
        # Transform text
        try:
            X_tfidf = self.vectorizer.transform([cleaned_text])
        except Exception as e:
            print(f"Error transforming text: {e}")
            # If vectorizer fails, use fallback
            return self.fallback_prediction(symptoms_text, top_k)
        
        # Get predictions from all models
        all_probas = []
        for name, model in self.model.items():
            try:
                proba = model.predict_proba(X_tfidf)[0]
                all_probas.append(proba)
            except Exception as e:
                print(f"Error with {name} model: {e}")
                continue
        
        if not all_probas:
            return self.fallback_prediction(symptoms_text, top_k)
        
        # Average probabilities
        avg_proba = np.mean(all_probas, axis=0)
        
        # Get top k predictions
        top_indices = np.argsort(avg_proba)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if avg_proba[idx] > 0.01:  # Only include if probability > 1%
                disease = self.label_encoder.inverse_transform([idx])[0]
                probability = avg_proba[idx] * 100
                
                # Get disease information
                disease_info = self.disease_database.get(disease, {})
                
                # Calculate symptom match
                user_symptoms = cleaned_text.split()
                disease_symptoms = disease_info.get('symptoms', [])
                
                if disease_symptoms:
                    matched_symptoms = set(user_symptoms).intersection(set(disease_symptoms))
                    symptom_match = (len(matched_symptoms) / len(disease_symptoms)) * 100
                else:
                    symptom_match = 0
                    matched_symptoms = set()
                
                # Determine model agreement
                model_agreement = self.calculate_model_agreement(all_probas, idx)
                
                results.append({
                    'disease': disease,
                    'probability': round(probability, 2),
                    'symptom_match': round(symptom_match, 1),
                    'matched_symptoms': list(matched_symptoms),
                    'severity': disease_info.get('severity', 2),
                    'urgency': disease_info.get('urgency', 'moderate'),
                    'body_system': disease_info.get('body_system', 'unknown'),
                    'key_symptoms': disease_symptoms[:5],
                    'model_agreement': model_agreement,
                    'confidence_interval': self.calculate_confidence_interval(probability),
                    'next_steps': self.get_recommendations(disease, probability),
                    'warning_signs': self.get_warning_signs(disease)
                })
        
        return results
    
    def calculate_model_agreement(self, all_probas: List[np.ndarray], idx: int) -> float:
        """Calculate agreement between models for a specific disease"""
        agreements = []
        for proba in all_probas:
            # Check if this model also has this disease in top 3
            top_3_indices = np.argsort(proba)[-3:][::-1]
            if idx in top_3_indices:
                agreements.append(1)
            else:
                agreements.append(0)
        
        return round((sum(agreements) / len(agreements)) * 100, 1)
    
    def calculate_confidence_interval(self, probability: float, margin: float = 5.0) -> Dict:
        """Calculate confidence interval for prediction"""
        return {
            'lower': max(0, round(probability - margin, 2)),
            'upper': min(100, round(probability + margin, 2)),
            'margin': margin
        }
    
    def get_recommendations(self, disease: str, probability: float) -> List[str]:
        """Get actionable recommendations"""
        recommendations = []
        
        # Based on probability
        if probability > 80:
            recommendations.append(f"High probability of {disease} - consult healthcare professional urgently")
        elif probability > 50:
            recommendations.append(f"Moderate probability - schedule doctor appointment within 48 hours")
        else:
            recommendations.append("Monitor symptoms and consult if they worsen")
        
        # General recommendations
        recommendations.append("Rest and maintain hydration")
        recommendations.append("Avoid self-medication without professional advice")
        recommendations.append("Monitor symptoms and note any changes")
        
        # Disease-specific recommendations
        if 'fever' in disease.lower() or 'flu' in disease.lower():
            recommendations.append("Monitor temperature regularly")
            recommendations.append("Consider fever-reducing medication if approved by doctor")
        
        if 'cough' in disease.lower() or 'respiratory' in disease.lower():
            recommendations.append("Use humidifier to ease breathing")
            recommendations.append("Avoid irritants like smoke or strong smells")
        
        return recommendations[:4]  # Return top 4 recommendations
    
    def get_warning_signs(self, disease: str) -> List[str]:
        """Get warning signs for specific diseases"""
        warning_signs = {
            'Heart Attack': ['Chest pain spreading to arms', 'Severe shortness of breath', 'Sudden dizziness'],
            'Pneumonia': ['High fever (>103°F)', 'Severe chest pain', 'Confusion or disorientation'],
            'COVID-19': ['Difficulty breathing', 'Persistent chest pain', 'Blue lips or face'],
            'Stroke': ['Sudden numbness', 'Confusion', 'Trouble speaking', 'Vision problems'],
            'Diabetes': ['Extreme thirst', 'Frequent urination', 'Blurred vision', 'Unexplained weight loss']
        }
        
        return warning_signs.get(disease, [])
    
    def fallback_prediction(self, symptoms_text: str, top_k: int = 3) -> List[Dict]:
        """Fallback prediction using rule-based matching"""
        print("Using fallback rule-based prediction...")
        
        cleaned_text = self.preprocess_text(symptoms_text)
        user_symptoms = cleaned_text.split()
        
        results = []
        for disease, info in self.disease_database.items():
            disease_symptoms = info.get('symptoms', [])
            
            if disease_symptoms:
                matched_symptoms = set(user_symptoms).intersection(set(disease_symptoms))
                if matched_symptoms:
                    match_percentage = (len(matched_symptoms) / len(disease_symptoms)) * 100
                    
                    if match_percentage > 20:  # At least 20% match
                        results.append({
                            'disease': disease,
                            'probability': round(match_percentage, 2),
                            'symptom_match': round(match_percentage, 1),
                            'matched_symptoms': list(matched_symptoms),
                            'severity': info.get('severity', 2),
                            'urgency': info.get('urgency', 'moderate'),
                            'body_system': info.get('body_system', 'unknown'),
                            'key_symptoms': disease_symptoms[:5],
                            'model_agreement': 100.0,  # Rule-based is always consistent
                            'confidence_interval': {'lower': max(0, match_percentage-10), 
                                                   'upper': min(100, match_percentage+10), 
                                                   'margin': 10},
                            'next_steps': self.get_recommendations(disease, match_percentage),
                            'warning_signs': self.get_warning_signs(disease),
                            'note': 'Using rule-based matching'
                        })
        
        # Sort and limit
        results.sort(key=lambda x: x['probability'], reverse=True)
        return results[:top_k]
    
    def analyze_symptoms(self, symptoms_text: str) -> Dict:
        """Main method to analyze symptoms and return comprehensive results"""
        start_time = datetime.now()
        
        print(f"Analyzing symptoms: {symptoms_text[:50]}...")
        
        # Get predictions
        predictions = self.predict_ensemble(symptoms_text, top_k=5)
        
        # Generate overall assessment
        assessment = self.generate_assessment(predictions)
        
        # Extract symptom information
        extracted_info = self.extract_symptom_info(symptoms_text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'results': predictions,
            'assessment': assessment,
            'extracted_info': extracted_info,
            'processing_time': round(processing_time, 2),
            'model_info': {
                'type': 'Ensemble (Random Forest, Gradient Boosting, Naive Bayes)',
                'diseases_covered': len(self.disease_database),
                'is_trained': self.is_trained
            },
            'timestamp': datetime.now().isoformat(),
            'disclaimer': 'This is for informational purposes only. Always consult a healthcare professional for medical advice.'
        }
    
    def extract_symptom_info(self, symptoms_text: str) -> Dict:
        """Extract and categorize symptoms from text"""
        cleaned_text = self.preprocess_text(symptoms_text)
        words = cleaned_text.split()
        
        # Common symptom categories
        symptom_categories = {
            'respiratory': ['cough', 'breath', 'chest', 'throat', 'nose', 'sneeze'],
            'fever': ['fever', 'temperature', 'chills', 'sweat'],
            'pain': ['pain', 'ache', 'sore', 'hurt'],
            'gastrointestinal': ['stomach', 'nausea', 'vomit', 'diarrhea', 'constipation'],
            'neurological': ['headache', 'dizziness', 'confusion', 'memory'],
            'general': ['tired', 'fatigue', 'weak', 'malaise']
        }
        
        extracted_symptoms = []
        categories_found = set()
        
        for word in words:
            for category, keywords in symptom_categories.items():
                if any(keyword in word for keyword in keywords):
                    extracted_symptoms.append(word)
                    categories_found.add(category)
                    break
        
        return {
            'raw_text': symptoms_text,
            'cleaned_text': cleaned_text,
            'extracted_symptoms': list(set(extracted_symptoms)),
            'categories': list(categories_found),
            'word_count': len(words)
        }
    
    def generate_assessment(self, predictions: List[Dict]) -> Dict:
        """Generate overall assessment based on predictions"""
        if not predictions:
            return {
                'risk_level': 'low',
                'urgency': 'routine',
                'recommendation': 'Symptoms not recognized. Please consult a healthcare professional.',
                'confidence': 'low',
                'primary_concern': 'Unknown'
            }
        
        top_prediction = predictions[0]
        
        # Determine risk level
        if top_prediction['severity'] >= 4 or top_prediction['probability'] > 80:
            risk_level = 'high'
            urgency = 'urgent'
        elif top_prediction['severity'] >= 3 or top_prediction['probability'] > 50:
            risk_level = 'medium'
            urgency = 'soon'
        else:
            risk_level = 'low'
            urgency = 'routine'
        
        # Determine confidence
        if top_prediction['probability'] > 70 and top_prediction['model_agreement'] > 80:
            confidence = 'high'
        elif top_prediction['probability'] > 40:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'risk_level': risk_level,
            'urgency': urgency,
            'confidence': confidence,
            'primary_concern': top_prediction['disease'],
            'top_probability': top_prediction['probability'],
            'recommendation': f"Based on analysis, the most likely condition is {top_prediction['disease']} ({top_prediction['probability']}% probability).",
            'action_required': self.get_action_required(risk_level, urgency)
        }
    
    def get_action_required(self, risk_level: str, urgency: str) -> str:
        """Get action required based on risk and urgency"""
        actions = {
            ('high', 'emergency'): 'Seek immediate medical attention. Call emergency services if severe symptoms.',
            ('high', 'urgent'): 'Consult healthcare professional within 24 hours.',
            ('medium', 'soon'): 'Schedule doctor appointment within 48-72 hours.',
            ('low', 'routine'): 'Monitor symptoms and consult if they persist beyond 3 days.'
        }
        
        return actions.get((risk_level, urgency), 'Consult healthcare professional for proper diagnosis.')

# Create singleton instance
symptom_checker = SymptomChecker()

# Initialize model on import
print("Initializing Symptom Checker...")
if not symptom_checker.is_trained:
    print("No pre-trained model found. Training new model...")
    symptom_checker.train_model()
else:
    print("Pre-trained model loaded successfully!")