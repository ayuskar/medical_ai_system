"""
Enhanced Sentiment Analyzer for Medical Reviews
Uses BERT model with medical-specific fine-tuning and fallback mechanisms
"""

import re
import json
import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from django.conf import settings

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Transformers library not available. Using rule-based analyzer only.")

class EnhancedSentimentAnalyzer:
    """
    Advanced sentiment analyzer for medical reviews with multiple analysis modes
    """
    
    def __init__(self):
        self.bert_model = None
        self.tokenizer = None
        self.rule_based_model = None
        self.medical_lexicon = self.create_medical_lexicon()
        self.model_loaded = False
        self.model_path = os.path.join(settings.BASE_DIR, 'ai_predictions', 'ml_models')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize analyzer
        self.initialize_analyzer()
    
    def create_medical_lexicon(self) -> Dict:
        """Create comprehensive medical sentiment lexicon"""
        return {
            'positive': {
                # Service quality
                'professional', 'excellent', 'outstanding', 'exceptional', 'superb',
                'caring', 'compassionate', 'empathetic', 'kind', 'patient', 'attentive',
                'thorough', 'detailed', 'comprehensive', 'meticulous', 'precise',
                
                # Medical outcomes
                'healed', 'recovered', 'better', 'improved', 'relieved', 'cured',
                'treated', 'diagnosed', 'prescribed', 'medicated', 'therapized',
                'successful', 'effective', 'efficient', 'timely', 'prompt',
                
                # Facility & environment
                'clean', 'hygienic', 'sterile', 'modern', 'advanced', 'equipped',
                'comfortable', 'welcoming', 'pleasant', 'spacious', 'organized',
                
                # Communication
                'explained', 'clarified', 'listened', 'understood', 'communicated',
                'clear', 'transparent', 'honest', 'truthful', 'informative',
                
                # Overall experience
                'satisfied', 'happy', 'pleased', 'grateful', 'thankful', 'impressed',
                'recommend', 'trust', 'confidence', 'reliable', 'dependable'
            },
            
            'negative': {
                # Service quality issues
                'unprofessional', 'rude', 'impolite', 'disrespectful', 'arrogant',
                'inattentive', 'negligent', 'careless', 'reckless', 'irresponsible',
                
                # Medical issues
                'misdiagnosed', 'misdiagnosis', 'undiagnosed', 'overdiagnosed',
                'overprescribed', 'underprescribed', 'wrong', 'incorrect', 'error',
                'mistake', 'failed', 'unsuccessful', 'ineffective', 'inefficient',
                
                # Pain and discomfort
                'painful', 'hurt', 'suffering', 'agony', 'torture', 'trauma',
                'uncomfortable', 'distressing', 'disturbing', 'frightening', 'scary',
                
                # Time and waiting
                'delayed', 'late', 'slow', 'waiting', 'postponed', 'cancelled',
                'rescheduled', 'bureaucratic', 'red tape', 'paperwork', 'hassle',
                
                # Communication issues
                'unclear', 'vague', 'confusing', 'misleading', 'deceptive',
                'ignored', 'dismissed', 'rushed', 'hurried', 'abrupt',
                
                # Cost and billing
                'expensive', 'costly', 'overcharged', 'overpriced', 'unaffordable',
                'billing', 'insurance', 'coverage', 'denied', 'rejected',
                
                # Overall dissatisfaction
                'dissatisfied', 'unhappy', 'disappointed', 'frustrated', 'angry',
                'upset', 'annoyed', 'regret', 'avoid', 'warning', 'complaint'
            },
            
            'medical_positive': {
                # Treatment success
                'treatment successful', 'surgery successful', 'operation successful',
                'medication effective', 'therapy helpful', 'rehabilitation complete',
                'pain relief', 'symptom relief', 'disease free', 'cancer free',
                'infection cleared', 'wound healed', 'fracture healed',
                
                # Professional competence
                'expert diagnosis', 'accurate diagnosis', 'correct treatment',
                'appropriate medication', 'skilled surgeon', 'experienced doctor',
                'knowledgeable staff', 'competent nurse', 'qualified specialist',
                
                # Patient experience
                'bedside manner', 'patient care', 'emotional support', 
                'family included', 'follow-up care', 'aftercare support'
            },
            
            'medical_negative': {
                # Medical complications
                'complication', 'side effect', 'adverse reaction', 'allergic reaction',
                'infection', 'bleeding', 'swelling', 'inflammation', 'fever',
                'nausea', 'vomiting', 'diarrhea', 'constipation', 'headache',
                'dizziness', 'fatigue', 'weakness', 'pain', 'discomfort',
                
                # Treatment failures
                'treatment failed', 'surgery failed', 'medication ineffective',
                'therapy unsuccessful', 'condition worsened', 'disease progressed',
                'relapse', 'recurrence', 'chronic', 'persistent', 'untreatable',
                
                # Diagnostic errors
                'false positive', 'false negative', 'missed diagnosis', 
                'delayed diagnosis', 'wrong diagnosis', 'incorrect test results'
            },
            
            'intensifiers': {
                'very', 'extremely', 'highly', 'absolutely', 'completely',
                'totally', 'utterly', 'exceptionally', 'incredibly', 'remarkably'
            },
            
            'negators': {
                'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'nobody',
                'neither', 'nor', 'hardly', 'scarcely', 'barely'
            }
        }
    
    def initialize_analyzer(self):
        """Initialize the sentiment analyzer with best available model"""
        print("Initializing Enhanced Sentiment Analyzer...")
        
        # Try to load BERT model first
        if HAS_TRANSFORMERS and self.load_bert_model():
            print("✓ BERT model loaded successfully")
            self.model_loaded = True
        else:
            print("⚠ Using rule-based sentiment analyzer")
            self.model_loaded = True  # Rule-based is always available
    
    def load_bert_model(self) -> bool:
        """Load pre-trained BERT model for sentiment analysis"""
        try:
            # Try to load from local cache first
            local_model_path = os.path.join(self.model_path, 'sentiment_model')
            
            if os.path.exists(local_model_path):
                print("Loading local BERT model...")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            else:
                print("Downloading BERT model...")
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Save for future use
                self.tokenizer.save_pretrained(local_model_path)
                self.bert_model.save_pretrained(local_model_path)
                print(f"Model saved to {local_model_path}")
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.cuda()
                print("Model moved to GPU")
            
            return True
            
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_with_bert(self, text: str) -> Dict:
        """Analyze sentiment using BERT model"""
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Truncate if too long
            if len(cleaned_text.split()) > 512:
                words = cleaned_text.split()[:512]
                cleaned_text = ' '.join(words)
            
            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get scores
            scores = predictions.cpu().numpy()[0]
            
            # Map to sentiment labels
            sentiment_labels = ['NEGATIVE', 'POSITIVE']
            best_index = np.argmax(scores)
            
            return {
                'label': sentiment_labels[best_index],
                'score': float(scores[best_index]),
                'confidence': float(scores[best_index]),
                'method': 'bert'
            }
            
        except Exception as e:
            print(f"Error in BERT analysis: {e}")
            return None
    
    def analyze_with_rules(self, text: str) -> Dict:
        """Advanced rule-based sentiment analysis for medical context"""
        text_lower = text.lower()
        cleaned_text = self.preprocess_text(text_lower)
        words = set(re.findall(r'\b\w+\b', cleaned_text))
        bigrams = set([' '.join(pair) for pair in zip(cleaned_text.split(), cleaned_text.split()[1:])])
        
        # Initialize scores
        positive_score = 0
        negative_score = 0
        medical_positive_score = 0
        medical_negative_score = 0
        
        # Check for bigrams (medical phrases)
        for bigram in bigrams:
            if bigram in self.medical_lexicon['medical_positive']:
                medical_positive_score += 2
            if bigram in self.medical_lexicon['medical_negative']:
                medical_negative_score += 2
        
        # Check individual words
        for word in words:
            # Check positive words
            if word in self.medical_lexicon['positive']:
                positive_score += 1
            # Check negative words
            if word in self.medical_lexicon['negative']:
                negative_score += 1
            # Check medical positive
            if word in self.medical_lexicon['medical_positive']:
                medical_positive_score += 1
            # Check medical negative
            if word in self.medical_lexicon['medical_negative']:
                medical_negative_score += 1
            # Check intensifiers
            if word in self.medical_lexicon['intensifiers']:
                # Check next word for sentiment
                idx = cleaned_text.split().index(word) if word in cleaned_text.split() else -1
                if idx != -1 and idx + 1 < len(cleaned_text.split()):
                    next_word = cleaned_text.split()[idx + 1]
                    if next_word in self.medical_lexicon['positive']:
                        positive_score += 0.5
                    if next_word in self.medical_lexicon['negative']:
                        negative_score += 0.5
        
        # Check for negators
        negator_pattern = r'\b(not|no|never|none)\b\s+\b(\w+)\b'
        matches = re.findall(negator_pattern, text_lower)
        for negator, target in matches:
            if target in self.medical_lexicon['positive']:
                negative_score += 1  # "not good" is negative
            if target in self.medical_lexicon['negative']:
                positive_score += 1  # "not bad" is positive
        
        # Calculate total scores with weights
        total_positive = positive_score + (medical_positive_score * 1.5)
        total_negative = negative_score + (medical_negative_score * 1.2)
        
        # Determine sentiment
        if total_positive > total_negative:
            confidence = min(0.9, 0.5 + (total_positive / (total_positive + total_negative + 1)) * 0.4)
            return {
                'label': 'POSITIVE',
                'score': round(confidence, 3),
                'confidence': round(confidence, 3),
                'method': 'rule_based',
                'details': {
                    'positive_score': total_positive,
                    'negative_score': total_negative,
                    'medical_terms': medical_positive_score + medical_negative_score
                }
            }
        elif total_negative > total_positive:
            confidence = min(0.9, 0.5 + (total_negative / (total_positive + total_negative + 1)) * 0.4)
            return {
                'label': 'NEGATIVE',
                'score': round(confidence, 3),
                'confidence': round(confidence, 3),
                'method': 'rule_based',
                'details': {
                    'positive_score': total_positive,
                    'negative_score': total_negative,
                    'medical_terms': medical_positive_score + medical_negative_score
                }
            }
        else:
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'confidence': 0.5,
                'method': 'rule_based',
                'details': {
                    'positive_score': total_positive,
                    'negative_score': total_negative,
                    'medical_terms': 0
                }
            }
    
    def analyze_emotion(self, text: str) -> Dict:
        """Detect emotional tone in medical reviews"""
        text_lower = text.lower()
        
        emotion_patterns = {
            'gratitude': [r'thank', r'grateful', r'appreciate', r'blessed'],
            'anger': [r'angry', r'furious', r'enraged', r'outraged'],
            'fear': [r'scared', r'afraid', r'fear', r'terrified', r'anxious'],
            'sadness': [r'sad', r'upset', r'disappointed', r'heartbroken'],
            'relief': [r'relieved', r'relief', r'peace', r'calm'],
            'trust': [r'trust', r'confident', r'faith', r'reliable'],
            'surprise': [r'surprised', r'shocked', r'amazed', r'astonished']
        }
        
        emotions_detected = {}
        for emotion, patterns in emotion_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text_lower))
            if count > 0:
                emotions_detected[emotion] = count
        
        # Find dominant emotion
        if emotions_detected:
            dominant_emotion = max(emotions_detected, key=emotions_detected.get)
            emotion_strength = emotions_detected[dominant_emotion] / len(text_lower.split()) * 10
        else:
            dominant_emotion = 'neutral'
            emotion_strength = 0
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_strength': round(min(1.0, emotion_strength), 2),
            'all_emotions': emotions_detected
        }
    
    def analyze_aspects(self, text: str) -> Dict:
        """Extract specific aspects mentioned in medical reviews"""
        aspects = {
            'doctor': ['doctor', 'physician', 'surgeon', 'specialist', 'dr'],
            'staff': ['nurse', 'staff', 'receptionist', 'assistant', 'technician'],
            'facility': ['hospital', 'clinic', 'office', 'room', 'facility', 'building'],
            'treatment': ['treatment', 'medication', 'therapy', 'surgery', 'procedure'],
            'waiting': ['wait', 'time', 'appointment', 'schedule', 'delay'],
            'cost': ['cost', 'price', 'bill', 'insurance', 'payment', 'expensive'],
            'communication': ['explain', 'talk', 'listen', 'understand', 'communicate']
        }
        
        text_lower = text.lower()
        aspects_detected = {}
        
        for aspect, keywords in aspects.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                aspects_detected[aspect] = count
        
        return aspects_detected
    
    def analyze_sentiment(self, text: str, detailed: bool = False) -> Dict:
        """
        Main method to analyze sentiment of medical review text
        """
        if not text or len(text.strip()) < 3:
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'confidence': 0.5,
                'method': 'none',
                'error': 'Text too short'
            }
        
        # Try BERT model first
        bert_result = None
        if self.bert_model is not None:
            bert_result = self.analyze_with_bert(text)
        
        # Always get rule-based result as fallback
        rule_result = self.analyze_with_rules(text)
        
        # Choose best result
        if bert_result and bert_result['confidence'] > rule_result['confidence']:
            result = bert_result
        else:
            result = rule_result
        
        # Add detailed analysis if requested
        if detailed:
            emotion_analysis = self.analyze_emotion(text)
            aspect_analysis = self.analyze_aspects(text)
            
            result.update({
                'emotion': emotion_analysis,
                'aspects': aspect_analysis,
                'text_length': len(text),
                'word_count': len(text.split()),
                'timestamp': datetime.now().isoformat()
            })
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts at once"""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def get_sentiment_summary(self, texts: List[str]) -> Dict:
        """Get summary statistics for multiple reviews"""
        if not texts:
            return {
                'total_reviews': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'average_score': 0.5,
                'sentiment_distribution': {}
            }
        
        results = self.analyze_batch(texts)
        
        positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        negative_count = sum(1 for r in results if r['label'] == 'NEGATIVE')
        neutral_count = sum(1 for r in results if r['label'] == 'NEUTRAL')
        
        average_score = sum(r['score'] for r in results) / len(results)
        
        return {
            'total_reviews': len(texts),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_percentage': round(positive_count / len(texts) * 100, 1),
            'negative_percentage': round(negative_count / len(texts) * 100, 1),
            'average_score': round(average_score, 3),
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
        }

# Singleton instance
sentiment_analyzer = EnhancedSentimentAnalyzer()