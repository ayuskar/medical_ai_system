import re
from transformers import pipeline
import joblib
import os
from django.conf import settings

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = None
        self.model_loaded = False
    
    def load_model(self):
        """Load pre-trained sentiment analysis model"""
        try:
            self.classifier = pipeline(
                'sentiment-analysis',
                model='distilbert-base-uncased-finetuned-sst-2-english',
                tokenizer='distilbert-base-uncased-finetuned-sst-2-english'
            )
            self.model_loaded = True
            print("Sentiment analysis model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            # Fallback to rule-based approach
            return self.setup_rule_based_analyzer()
    
    def setup_rule_based_analyzer(self):
        """Setup rule-based sentiment analysis as fallback"""
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'outstanding', 'fantastic',
            'professional', 'caring', 'helpful', 'kind', 'patient', 'knowledgeable', 'thorough',
            'recommend', 'satisfied', 'happy', 'pleased', 'impressed', 'grateful', 'thankful',
            'comfortable', 'trust', 'confident', 'clean', 'modern', 'efficient', 'quick',
            'listened', 'understood', 'explained', 'clear', 'detailed', 'comprehensive'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'rude', 'unprofessional', 'disappointed',
            'worst', 'waste', 'poor', 'slow', 'late', 'dirty', 'messy', 'disorganized',
            'uncomfortable', 'painful', 'scared', 'nervous', 'confused', 'frustrated',
            'angry', 'upset', 'annoyed', 'dissatisfied', 'unhappy', 'regret', 'avoid',
            'rushed', 'ignored', 'dismissed', 'unclear', 'vague', 'misleading', 'wrong'
        }
        
        self.medical_positive = {
            'cured', 'healed', 'recovered', 'better', 'improved', 'relieved', 'helped',
            'treated', 'diagnosed', 'prescribed', 'medication', 'therapy', 'surgery',
            'procedure', 'operation', 'recovery', 'healing', 'treatment', 'care'
        }
        
        self.medical_negative = {
            'pain', 'hurt', 'suffering', 'sick', 'ill', 'disease', 'infection', 'virus',
            'bacteria', 'fever', 'headache', 'nausea', 'vomiting', 'diarrhea', 'bleeding',
            'swelling', 'inflammation', 'complication', 'side effect', 'allergic', 'reaction'
        }
        
        self.model_loaded = True
        print("Rule-based sentiment analyzer setup completed!")
        return True
    
    def analyze_sentiment_rule_based(self, text):
        """Rule-based sentiment analysis for medical reviews"""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))
        medical_positive_count = len(words.intersection(self.medical_positive))
        medical_negative_count = len(words.intersection(self.medical_negative))
        
        # Adjust weights for medical context
        total_positive = positive_count + (medical_positive_count * 0.5)
        total_negative = negative_count + (medical_negative_count * 0.3)
        
        if total_positive > total_negative:
            score = min(0.9, 0.5 + (total_positive / (total_positive + total_negative + 1)) * 0.4)
            return {'label': 'POSITIVE', 'score': round(score, 3)}
        elif total_negative > total_positive:
            score = min(0.9, 0.5 + (total_negative / (total_positive + total_negative + 1)) * 0.4)
            return {'label': 'NEGATIVE', 'score': round(score, 3)}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of medical review text"""
        if not self.model_loaded:
            if not self.load_model():
                return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            # Clean text
            text = re.sub(r'[^\w\s.,!?]', '', text)
            
            if hasattr(self, 'classifier') and self.classifier:
                # Use transformer model
                result = self.classifier(text[:512])[0]  # Limit text length
                return {
                    'label': result['label'],
                    'score': round(result['score'], 3)
                }
            else:
                # Use rule-based approach
                return self.analyze_sentiment_rule_based(text)
                
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Fallback to rule-based
            return self.analyze_sentiment_rule_based(text)

# Singleton instance
sentiment_analyzer = SentimentAnalyzer()