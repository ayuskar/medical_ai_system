import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re
from django.conf import settings

class SymptomChecker:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.disease_database = self.create_comprehensive_database()
    
    def create_comprehensive_database(self):
        """Create a comprehensive symptom-disease database"""
        disease_data = {
            # Respiratory Diseases
            'Common Cold': ['runny nose', 'sneezing', 'sore throat', 'cough', 'congestion', 'mild fever', 'headache'],
            'Influenza (Flu)': ['fever', 'chills', 'muscle aches', 'cough', 'sore throat', 'headache', 'fatigue', 'runny nose'],
            'COVID-19': ['fever', 'dry cough', 'tiredness', 'loss of taste', 'loss of smell', 'sore throat', 'headache', 'diarrhea'],
            'Pneumonia': ['fever', 'chills', 'cough with phlegm', 'chest pain', 'shortness of breath', 'fatigue', 'sweating'],
            'Bronchitis': ['cough', 'mucus production', 'fatigue', 'shortness of breath', 'slight fever', 'chest discomfort'],
            'Asthma': ['wheezing', 'shortness of breath', 'chest tightness', 'coughing', 'difficulty breathing'],
            'Tuberculosis': ['cough lasting weeks', 'chest pain', 'coughing blood', 'fatigue', 'fever', 'night sweats', 'weight loss'],
            
            # Cardiovascular Diseases
            'Heart Attack': ['chest pain', 'shortness of breath', 'nausea', 'lightheadedness', 'pain in arms', 'sweating'],
            'Hypertension': ['headaches', 'shortness of breath', 'nosebleeds', 'flushing', 'dizziness', 'chest pain'],
            'Heart Failure': ['shortness of breath', 'fatigue', 'swollen legs', 'rapid heartbeat', 'persistent cough'],
            'Arrhythmia': ['palpitations', 'dizziness', 'fainting', 'shortness of breath', 'chest pain', 'fatigue'],
            
            # Gastrointestinal Diseases
            'Gastroenteritis': ['diarrhea', 'vomiting', 'abdominal pain', 'nausea', 'fever', 'headache'],
            'Food Poisoning': ['nausea', 'vomiting', 'watery diarrhea', 'abdominal pain', 'fever', 'headache'],
            'Irritable Bowel Syndrome': ['abdominal pain', 'bloating', 'gas', 'diarrhea', 'constipation', 'mucus in stool'],
            'Appendicitis': ['abdominal pain', 'nausea', 'vomiting', 'loss of appetite', 'fever', 'constipation', 'diarrhea'],
            'Gallstones': ['abdominal pain', 'back pain', 'nausea', 'vomiting', 'bloating', 'indigestion'],
            'GERD': ['heartburn', 'chest pain', 'difficulty swallowing', 'regurgitation', 'sore throat', 'chronic cough'],
            
            # Neurological Diseases
            'Migraine': ['headache', 'nausea', 'sensitivity to light', 'sensitivity to sound', 'aura', 'vomiting'],
            'Tension Headache': ['headache', 'pressure around forehead', 'tenderness in scalp', 'neck pain'],
            'Stroke': ['face drooping', 'arm weakness', 'speech difficulty', 'headache', 'dizziness', 'vision problems'],
            'Epilepsy': ['seizures', 'temporary confusion', 'staring spells', 'uncontrollable movements', 'loss of consciousness'],
            
            # Musculoskeletal Diseases
            'Arthritis': ['joint pain', 'stiffness', 'swelling', 'redness', 'decreased range of motion'],
            'Osteoporosis': ['back pain', 'loss of height', 'stooped posture', 'bone fracture'],
            'Fibromyalgia': ['widespread pain', 'fatigue', 'sleep problems', 'memory issues', 'mood issues'],
            
            # Endocrine Diseases
            'Diabetes': ['increased thirst', 'frequent urination', 'hunger', 'fatigue', 'blurred vision', 'slow healing'],
            'Thyroid Disorder': ['fatigue', 'weight changes', 'mood changes', 'hair loss', 'temperature sensitivity'],
            'Adrenal Fatigue': ['fatigue', 'body aches', 'nervousness', 'sleep disturbances', 'digestive problems'],
            
            # Dermatological Diseases
            'Eczema': ['itchy skin', 'redness', 'dry skin', 'cracks', 'blisters', 'swelling'],
            'Psoriasis': ['red patches', 'silvery scales', 'dry skin', 'itching', 'burning', 'thickened nails'],
            'Acne': ['pimples', 'blackheads', 'whiteheads', 'cystic lesions', 'redness', 'inflammation'],
            
            # Infectious Diseases
            'Malaria': ['fever', 'chills', 'headache', 'nausea', 'vomiting', 'muscle pain', 'sweating'],
            'Dengue Fever': ['high fever', 'headache', 'pain behind eyes', 'joint pain', 'rash', 'bleeding'],
            'Hepatitis': ['fatigue', 'nausea', 'abdominal pain', 'loss of appetite', 'dark urine', 'jaundice'],
            
            # Mental Health
            'Depression': ['sadness', 'loss of interest', 'fatigue', 'sleep changes', 'appetite changes', 'concentration issues'],
            'Anxiety Disorder': ['worry', 'restlessness', 'fatigue', 'concentration problems', 'irritability', 'sleep problems'],
            'Bipolar Disorder': ['mood swings', 'energy changes', 'sleep pattern changes', 'behavior changes'],
            
            # Eye Diseases
            'Conjunctivitis': ['red eyes', 'itching', 'tearing', 'discharge', 'gritty feeling', 'swelling'],
            'Cataracts': ['cloudy vision', 'difficulty seeing at night', 'sensitivity to light', 'seeing halos'],
            'Glaucoma': ['eye pain', 'blurred vision', 'headache', 'nausea', 'seeing halos around lights'],
            
            # Ear Diseases
            'Otitis Media': ['ear pain', 'fever', 'hearing loss', 'ear drainage', 'irritability'],
            'Tinnitus': ['ringing in ears', 'buzzing', 'hissing', 'clicking', 'roaring sounds'],
            
            # Kidney Diseases
            'Kidney Stones': ['severe pain', 'painful urination', 'blood in urine', 'nausea', 'vomiting', 'fever'],
            'Kidney Infection': ['fever', 'back pain', 'frequent urination', 'burning urination', 'nausea'],
            
            # Cancer (for awareness)
            'Lung Cancer': ['persistent cough', 'chest pain', 'shortness of breath', 'coughing blood', 'hoarseness', 'weight loss'],
            'Breast Cancer': ['breast lump', 'breast pain', 'nipple discharge', 'skin changes', 'swelling'],
            'Skin Cancer': ['skin changes', 'new mole', 'changing mole', 'skin sore that doesnt heal', 'redness'],
        }
        
        return disease_data
    
    def generate_training_data(self):
        """Generate comprehensive training data from disease database"""
        expanded_data = []
        
        for disease, symptoms in self.disease_database.items():
            # Create multiple variations for each disease
            for _ in range(50):
                # Shuffle symptoms and take a random subset
                np.random.shuffle(symptoms)
                num_symptoms = np.random.randint(2, len(symptoms) + 1)
                selected_symptoms = symptoms[:num_symptoms]
                
                # Create different text variations
                symptom_text_variations = [
                    ' '.join(selected_symptoms),
                    'having ' + ' and '.join(selected_symptoms),
                    'symptoms include ' + ', '.join(selected_symptoms),
                    'experiencing ' + ' with '.join(selected_symptoms),
                ]
                
                for symptom_text in symptom_text_variations:
                    expanded_data.append({
                        'symptoms': symptom_text,
                        'disease': disease
                    })
        
        return pd.DataFrame(expanded_data)
    
    def train_model(self):
        """Train the symptom classifier"""
        print("Generating training data...")
        df = self.generate_training_data()
        print(f"Generated {len(df)} training samples")
        
        # Initialize and fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            stop_words='english',
            ngram_range=(1, 2)  # Include single words and bigrams
        )
        X = self.vectorizer.fit_transform(df['symptoms'])
        
        # Initialize and fit label encoder
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['disease'])
        
        # Train model
        self.model = MultinomialNB(alpha=0.1)
        self.model.fit(X, y)
        
        # Save components
        model_dir = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models')
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_dir, 'symptom_model.pkl'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'symptom_vectorizer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'symptom_encoder.pkl'))
        
        print("Symptom checker model trained successfully!")
        return True
    
    def load_model(self):
        """Load trained model components"""
        try:
            model_dir = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models')
            self.model = joblib.load(os.path.join(model_dir, 'symptom_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(model_dir, 'symptom_vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'symptom_encoder.pkl'))
            print("Symptom checker model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading symptom model: {e}")
            return False
    
    def preprocess_symptoms(self, symptoms_text):
        """Preprocess and clean symptom text"""
        # Convert to lowercase
        text = symptoms_text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Expand common abbreviations
        abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'temp': 'temperature',
            'fever': 'fever',
            'cough': 'cough',
            'pain': 'pain',
            'headache': 'headache',
            'nausea': 'nausea',
            'vomiting': 'vomiting',
            'diarrhea': 'diarrhea',
            'fatigue': 'fatigue',
            'dizziness': 'dizziness',
        }
        
        words = text.split()
        expanded_words = [abbreviations.get(word, word) for word in words]
        return ' '.join(expanded_words)
    
    def find_similar_diseases(self, symptoms_list):
        """Find diseases with matching symptoms using rule-based approach"""
        similar_diseases = []
        symptoms_set = set(symptoms_list)
        
        for disease, disease_symptoms in self.disease_database.items():
            disease_symptoms_set = set(disease_symptoms)
            matching_symptoms = symptoms_set.intersection(disease_symptoms_set)
            
            if matching_symptoms:
                match_percentage = len(matching_symptoms) / len(disease_symptoms_set) * 100
                if match_percentage > 20:  # At least 20% symptom match
                    similar_diseases.append({
                        'disease': disease,
                        'matching_symptoms': list(matching_symptoms),
                        'match_percentage': round(match_percentage, 1),
                        'all_symptoms': disease_symptoms
                    })
        
        # Sort by match percentage
        similar_diseases.sort(key=lambda x: x['match_percentage'], reverse=True)
        return similar_diseases[:5]  # Return top 5 matches
    
    def predict(self, symptoms_text):
        """Predict possible conditions from symptoms using both ML and rule-based approaches"""
        if not self.model:
            if not self.load_model():
                print("Training new symptom model...")
                self.train_model()
        
        # Preprocess symptoms
        cleaned_symptoms = self.preprocess_symptoms(symptoms_text)
        
        # ML-based prediction
        ml_results = []
        try:
            symptoms_tfidf = self.vectorizer.transform([cleaned_symptoms])
            probabilities = self.model.predict_proba(symptoms_tfidf)[0]
            
            # Get top 5 predictions
            top_5_indices = np.argsort(probabilities)[-5:][::-1]
            
            for idx in top_5_indices:
                if probabilities[idx] > 0.01:  # Only include if probability > 1%
                    ml_results.append({
                        'disease': self.label_encoder.inverse_transform([idx])[0],
                        'probability': round(probabilities[idx] * 100, 2),
                        'type': 'ml_prediction'
                    })
        except Exception as e:
            print(f"ML prediction error: {e}")
        
        # Rule-based prediction
        symptoms_list = cleaned_symptoms.split()
        rule_based_results = self.find_similar_diseases(symptoms_list)
        
        # Combine and deduplicate results
        all_results = []
        seen_diseases = set()
        
        # Add ML results first
        for result in ml_results:
            if result['disease'] not in seen_diseases:
                all_results.append(result)
                seen_diseases.add(result['disease'])
        
        # Add rule-based results
        for result in rule_based_results:
            if result['disease'] not in seen_diseases:
                all_results.append({
                    'disease': result['disease'],
                    'probability': result['match_percentage'],
                    'type': 'rule_based',
                    'matching_symptoms': result['matching_symptoms']
                })
                seen_diseases.add(result['disease'])
        
        # Sort by probability and return top 5
        all_results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Add severity and recommendation
        for result in all_results[:5]:
            result['severity'] = self.assess_severity(result['disease'])
            result['recommendation'] = self.get_recommendation(result['disease'], result['severity'])
        
        return all_results[:5]
    
    def assess_severity(self, disease):
        """Assess disease severity for emergency guidance"""
        emergency_conditions = [
            'Heart Attack', 'Stroke', 'Appendicitis', 'Meningitis', 'Pneumonia',
            'Kidney Stones', 'Gallstones', 'Asthma Attack', 'COVID-19 Severe'
        ]
        
        urgent_conditions = [
            'Pneumonia', 'Bronchitis', 'Kidney Infection', 'Hepatitis',
            'Diabetes Complications', 'Hypertension Crisis'
        ]
        
        if disease in emergency_conditions:
            return 'emergency'
        elif disease in urgent_conditions:
            return 'urgent'
        else:
            return 'routine'
    
    def get_recommendation(self, disease, severity):
        """Get appropriate recommendation based on disease and severity"""
        if severity == 'emergency':
            return "Seek immediate medical attention. Go to emergency room or call emergency services."
        elif severity == 'urgent':
            return "Consult a healthcare professional within 24-48 hours. Monitor symptoms closely."
        else:
            return "Schedule an appointment with your doctor. Rest and monitor symptoms."

# Singleton instance
symptom_checker = SymptomChecker()