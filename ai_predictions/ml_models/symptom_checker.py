import pandas as pd
import numpy as np
import re
import joblib
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from django.conf import settings
from django.core.cache import cache

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier

# NLP Libraries
import spacy
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Deep Learning (optional)
try:
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertForSequenceClassification
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class AdvancedSymptomChecker:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.encoders = {}
        self.scalers = {}
        self.nlp = None
        self.lemmatizer = WordNetLemmatizer()
        self.disease_knowledge_base = self.create_knowledge_base()
        self.initialize_nlp()
        self.training_history = []
        
    def initialize_nlp(self):
        """Initialize NLP processor"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def create_knowledge_base(self):
        """Create comprehensive medical knowledge base"""
        return {
            # Disease data structure
            'diseases': {
                'Influenza': {
                    'symptoms': ['fever', 'cough', 'sore throat', 'runny nose', 'body aches', 'headache', 'chills', 'fatigue'],
                    'severity': 3,
                    'urgency': 'moderate',
                    'duration': '3-7 days',
                    'risk_factors': ['age >65', 'pregnancy', 'chronic conditions'],
                    'complications': ['pneumonia', 'sinus infections'],
                    'icd10': 'J11'
                },
                # ... (expand with more detailed diseases)
            },
            'symptom_hierarchy': {
                'fever': {'type': 'vital_sign', 'severity_scale': 1-10},
                'cough': {'type': 'respiratory', 'subtypes': ['dry', 'productive']},
                # ... hierarchical structure
            },
            'body_systems': {
                'respiratory': ['cough', 'shortness of breath', 'chest pain'],
                'cardiovascular': ['chest pain', 'palpitations', 'dizziness'],
                # ... all body systems
            }
        }
    
    def preprocess_text_advanced(self, text: str) -> Dict:
        """Advanced NLP preprocessing with medical context"""
        doc = self.nlp(text.lower())
        
        extracted = {
            'tokens': [],
            'entities': [],
            'symptoms': [],
            'duration': None,
            'severity': None,
            'body_parts': [],
            'qualifiers': []
        }
        
        # Extract medical entities and symptoms
        for token in doc:
            lemma = self.lemmatizer.lemmatize(token.text)
            extracted['tokens'].append(lemma)
            
            # Detect symptom patterns
            if lemma in self.disease_knowledge_base['symptom_hierarchy']:
                extracted['symptoms'].append(lemma)
            
            # Extract duration (e.g., "for 3 days")
            if token.like_num and token.nbor(1).text in ['days', 'hours', 'weeks', 'months']:
                extracted['duration'] = f"{token.text} {token.nbor(1).text}"
            
            # Extract severity qualifiers
            if lemma in ['mild', 'moderate', 'severe', 'extreme']:
                extracted['severity'] = lemma
        
        # Named Entity Recognition for body parts
        for ent in doc.ents:
            if ent.label_ in ['BODY_PART', 'ORG']:
                extracted['body_parts'].append(ent.text)
        
        return extracted
    
    def build_training_dataset(self):
        """Build comprehensive training dataset from multiple sources"""
        # 1. Generate synthetic data
        synthetic_data = self.generate_synthetic_data()
        
        # 2. Load real-world datasets if available
        real_data = self.load_medical_datasets()
        
        # 3. Combine and augment
        df = pd.concat([synthetic_data, real_data], ignore_index=True)
        
        # 4. Data augmentation
        df = self.augment_data(df)
        
        return df
    
    def generate_synthetic_data(self):
        """Generate realistic synthetic medical data"""
        diseases = list(self.disease_knowledge_base['diseases'].keys())
        data = []
        
        for disease in diseases:
            symptoms = self.disease_knowledge_base['diseases'][disease]['symptoms']
            
            # Create multiple variations
            for i in range(100):
                # Random symptom combination
                num_symptoms = np.random.randint(2, min(6, len(symptoms)) + 1)
                selected_symptoms = np.random.choice(symptoms, num_symptoms, replace=False)
                
                # Create natural language descriptions
                descriptions = [
                    f"I have been experiencing {', '.join(selected_symptoms[:-1])} and {selected_symptoms[-1]}",
                    f"Symptoms include: {', '.join(selected_symptoms)}",
                    f"Feeling {selected_symptoms[0]} along with {', '.join(selected_symptoms[1:])}",
                    f"Patient presents with {', '.join(selected_symptoms)}",
                    f"Complaining of {selected_symptoms[0]}, {selected_symptoms[1]} for past few days"
                ]
                
                for desc in descriptions:
                    # Add duration and severity variations
                    durations = ['', ' for 2 days', ' since yesterday', ' for a week']
                    severities = ['', ' mildly', ' severely', ' moderately']
                    
                    for dur in durations:
                        for sev in severities:
                            text = desc + sev + dur
                            data.append({
                                'text': text,
                                'disease': disease,
                                'symptoms': '|'.join(selected_symptoms),
                                'severity': sev.strip() or 'moderate',
                                'duration': dur.replace(' for ', '') or 'unknown'
                            })
        
        return pd.DataFrame(data)
    
    def load_medical_datasets(self):
        """Load real medical datasets if available"""
        datasets = []
        
        # Try to load public datasets
        dataset_paths = [
            'data/symptom_disease.csv',
            'data/medical_texts.csv',
            'data/patient_records.csv'
        ]
        
        for path in dataset_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    datasets.append(df)
                except:
                    continue
        
        return pd.concat(datasets, ignore_index=True) if datasets else pd.DataFrame()
    
    def augment_data(self, df):
        """Apply data augmentation techniques"""
        augmented = []
        
        for _, row in df.iterrows():
            text = row['text']
            
            # Synonym replacement
            augmented_text = self.synonym_replacement(text)
            augmented.append({**row.to_dict(), 'text': augmented_text})
            
            # Back translation simulation
            if np.random.random() > 0.7:
                back_translated = self.simulate_back_translation(text)
                augmented.append({**row.to_dict(), 'text': back_translated})
            
            # Contextual augmentation
            if np.random.random() > 0.5:
                contextual = self.add_context(text, row['disease'])
                augmented.append({**row.to_dict(), 'text': contextual})
        
        return pd.DataFrame(augmented)
    
    def train_ensemble_model(self):
        """Train ensemble of models for better accuracy"""
        print("Building training dataset...")
        df = self.build_training_dataset()
        
        if len(df) < 100:
            print("Insufficient data, generating more...")
            df = self.generate_synthetic_data()
        
        print(f"Training on {len(df)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['disease'], test_size=0.2, random_state=42
        )
        
        # Feature engineering
        print("Performing feature engineering...")
        X_tfidf, X_embeddings = self.extract_features(X_train)
        X_test_tfidf, X_test_embeddings = self.extract_features(X_test, train=False)
        
        # Encode labels
        self.encoders['disease'] = LabelEncoder()
        y_train_encoded = self.encoders['disease'].fit_transform(y_train)
        y_test_encoded = self.encoders['disease'].transform(y_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        print("Training ensemble models...")
        for name, model in models.items():
            print(f"Training {name}...")
            if name in ['neural_network', 'svm']:
                X_train_combined = np.hstack([X_tfidf.toarray(), X_embeddings])
                X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_embeddings])
                model.fit(X_train_combined, y_train_encoded)
                
                # Evaluate
                y_pred = model.predict(X_test_combined)
                accuracy = accuracy_score(y_test_encoded, y_pred)
                print(f"{name} accuracy: {accuracy:.4f}")
            else:
                model.fit(X_tfidf, y_train_encoded)
                y_pred = model.predict(X_test_tfidf)
                accuracy = accuracy_score(y_test_encoded, y_pred)
                print(f"{name} accuracy: {accuracy:.4f}")
            
            self.models[name] = model
        
        # Create voting classifier
        print("Creating ensemble voting classifier...")
        estimators = [(name, model) for name, model in self.models.items()]
        self.models['ensemble'] = VotingClassifier(estimators=estimators, voting='soft')
        self.models['ensemble'].fit(
            np.hstack([X_tfidf.toarray(), X_embeddings]), 
            y_train_encoded
        )
        
        # Final evaluation
        y_pred_ensemble = self.models['ensemble'].predict(
            np.hstack([X_test_tfidf.toarray(), X_test_embeddings])
        )
        ensemble_accuracy = accuracy_score(y_test_encoded, y_pred_ensemble)
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        # Save models
        self.save_models()
        
        return ensemble_accuracy
    
    def extract_features(self, texts, train=True):
        """Extract multiple feature types"""
        # TF-IDF features
        if train:
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2
            )
            X_tfidf = self.vectorizers['tfidf'].fit_transform(texts)
        else:
            X_tfidf = self.vectorizers['tfidf'].transform(texts)
        
        # Embedding features (simplified)
        X_embeddings = np.zeros((len(texts), 100))
        for i, text in enumerate(texts):
            tokens = text.split()
            if tokens:
                # Simple average embedding (in real implementation, use Word2Vec/BERT)
                X_embeddings[i] = np.random.randn(100)  # Placeholder
        
        return X_tfidf, X_embeddings
    
    def predict_with_explanations(self, symptoms_text: str) -> Dict:
        """Advanced prediction with explanations and confidence scores"""
        # Preprocess
        processed = self.preprocess_text_advanced(symptoms_text)
        
        # Extract features
        X_tfidf = self.vectorizers['tfidf'].transform([symptoms_text])
        X_embeddings = np.random.randn(1, 100)  # Placeholder
        X_combined = np.hstack([X_tfidf.toarray(), X_embeddings])
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_combined)[0]
                top_3_idx = np.argsort(proba)[-3:][::-1]
                predictions[name] = {
                    'diseases': self.encoders['disease'].inverse_transform(top_3_idx),
                    'probabilities': proba[top_3_idx]
                }
        
        # Ensemble prediction
        ensemble_proba = self.models['ensemble'].predict_proba(X_combined)[0]
        top_5_idx = np.argsort(ensemble_proba)[-5:][::-1]
        
        results = []
        for idx in top_5_idx:
            disease = self.encoders['disease'].inverse_transform([idx])[0]
            probability = ensemble_proba[idx] * 100
            
            # Get detailed disease info
            disease_info = self.disease_knowledge_base['diseases'].get(disease, {})
            
            # Calculate symptom match percentage
            symptom_match = self.calculate_symptom_match(
                processed['symptoms'], 
                disease_info.get('symptoms', [])
            )
            
            # Generate differential diagnosis
            differential = self.generate_differential_diagnosis(disease, processed['symptoms'])
            
            results.append({
                'disease': disease,
                'probability': round(probability, 2),
                'symptom_match': symptom_match,
                'severity': disease_info.get('severity', 2),
                'urgency': disease_info.get('urgency', 'moderate'),
                'key_symptoms': disease_info.get('symptoms', [])[:5],
                'risk_factors': disease_info.get('risk_factors', []),
                'complications': disease_info.get('complications', []),
                'differential_diagnosis': differential,
                'next_steps': self.generate_next_steps(disease, probability, processed),
                'confidence_interval': self.calculate_confidence_interval(probability),
                'model_agreement': self.calculate_model_agreement(predictions, disease)
            })
        
        # Add overall assessment
        assessment = self.generate_overall_assessment(results, processed)
        
        return {
            'results': results,
            'assessment': assessment,
            'extracted_info': processed,
            'model_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_symptom_match(self, user_symptoms: List, disease_symptoms: List) -> float:
        """Calculate percentage of symptom match"""
        if not disease_symptoms:
            return 0.0
        
        matched = len(set(user_symptoms).intersection(set(disease_symptoms)))
        return round((matched / len(disease_symptoms)) * 100, 1)
    
    def generate_differential_diagnosis(self, primary_disease: str, symptoms: List) -> List[Dict]:
        """Generate list of similar conditions to consider"""
        differentials = []
        
        for disease, info in self.disease_knowledge_base['diseases'].items():
            if disease != primary_disease:
                common_symptoms = len(set(symptoms).intersection(set(info['symptoms'])))
                if common_symptoms > 0:
                    similarity = common_symptoms / len(info['symptoms']) * 100
                    if similarity > 30:
                        differentials.append({
                            'disease': disease,
                            'similarity': round(similarity, 1),
                            'distinguishing_features': list(set(info['symptoms']) - set(symptoms))
                        })
        
        return sorted(differentials, key=lambda x: x['similarity'], reverse=True)[:3]
    
    def generate_next_steps(self, disease: str, probability: float, processed_info: Dict) -> List[str]:
        """Generate actionable next steps"""
        steps = []
        
        if probability > 80:
            steps.append(f"High probability of {disease} - consult healthcare professional urgently")
        elif probability > 50:
            steps.append(f"Moderate probability - schedule doctor appointment within 48 hours")
        else:
            steps.append("Low probability - monitor symptoms and consult if they worsen")
        
        # Add specific recommendations
        severity = processed_info.get('severity')
        if severity == 'severe':
            steps.append("Consider visiting emergency department if symptoms worsen")
        
        if 'fever' in processed_info['symptoms'] and processed_info.get('duration'):
            steps.append("Monitor temperature every 4 hours")
        
        steps.append("Rest and maintain hydration")
        steps.append("Avoid self-medication without professional advice")
        
        return steps
    
    def calculate_confidence_interval(self, probability: float) -> Dict:
        """Calculate confidence interval for prediction"""
        margin = 5  # ±5% margin
        return {
            'lower': max(0, round(probability - margin, 2)),
            'upper': min(100, round(probability + margin, 2)),
            'margin': margin
        }
    
    def calculate_model_agreement(self, predictions: Dict, disease: str) -> float:
        """Calculate agreement between different models"""
        if not predictions:
            return 100.0
        
        agreement_count = 0
        total_models = len(predictions)
        
        for model_pred in predictions.values():
            if disease in model_pred['diseases'][:3]:  # If in top 3
                agreement_count += 1
        
        return round((agreement_count / total_models) * 100, 1)
    
    def generate_overall_assessment(self, results: List[Dict], processed_info: Dict) -> Dict:
        """Generate overall assessment with risk analysis"""
        top_result = results[0] if results else None
        
        assessment = {
            'risk_level': 'low',
            'urgency': 'routine',
            'recommendation': 'Monitor symptoms',
            'warning_signs': [],
            'when_to_seek_help': 'If symptoms persist beyond 3 days',
            'self_care_tips': []
        }
        
        if top_result:
            # Determine risk level
            if top_result['probability'] > 80 or top_result['severity'] >= 4:
                assessment['risk_level'] = 'high'
                assessment['urgency'] = 'urgent'
                assessment['recommendation'] = 'Seek medical attention promptly'
            elif top_result['probability'] > 50:
                assessment['risk_level'] = 'medium'
                assessment['urgency'] = 'soon'
                assessment['recommendation'] = 'Schedule doctor appointment'
            
            # Add warning signs
            warning_symptoms = ['chest pain', 'shortness of breath', 'severe headache', 'high fever']
            for symptom in warning_symptoms:
                if symptom in processed_info['symptoms']:
                    assessment['warning_signs'].append(symptom)
            
            # Add self-care tips
            assessment['self_care_tips'] = [
                "Get adequate rest",
                "Stay hydrated",
                "Monitor symptoms",
                "Keep a symptom diary"
            ]
        
        return assessment
    
    def save_models(self):
        """Save trained models"""
        model_dir = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models')
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(model_dir, f'{name}_model.pkl'))
        
        joblib.dump(self.vectorizers['tfidf'], os.path.join(model_dir, 'vectorizer.pkl'))
        joblib.dump(self.encoders['disease'], os.path.join(model_dir, 'encoder.pkl'))
        
        # Save knowledge base
        with open(os.path.join(model_dir, 'knowledge_base.json'), 'w') as f:
            json.dump(self.disease_knowledge_base, f, indent=2)
    
    def load_models(self):
        """Load trained models"""
        model_dir = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models')
        
        try:
            # Load models
            for name in ['random_forest', 'gradient_boosting', 'svm', 'neural_network', 'ensemble']:
                path = os.path.join(model_dir, f'{name}_model.pkl')
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
            
            # Load preprocessing objects
            self.vectorizers['tfidf'] = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
            self.encoders['disease'] = joblib.load(os.path.join(model_dir, 'encoder.pkl'))
            
            # Load knowledge base
            with open(os.path.join(model_dir, 'knowledge_base.json'), 'r') as f:
                self.disease_knowledge_base = json.load(f)
            
            print("All models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Singleton instance
advanced_symptom_checker = AdvancedSymptomChecker()