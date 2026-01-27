#!/usr/bin/env python
"""
Initialization script for Symptom Checker
Run this script once to initialize and train the model
"""

import os
import sys
import django

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project_name.settings')
django.setup()

from ai_predictions.ml_models.symptom_checker import symptom_checker

def main():
    print("=" * 60)
    print("Symptom Checker Initialization Script")
    print("=" * 60)
    
    print("\n1. Checking current status...")
    if symptom_checker.is_trained:
        print("✓ Model is already trained and loaded")
        print(f"  Diseases in database: {len(symptom_checker.disease_database)}")
    else:
        print("✗ Model is not trained")
    
    print("\n2. Testing model prediction...")
    test_symptoms = "I have fever and cough with body aches"
    
    try:
        result = symptom_checker.analyze_symptoms(test_symptoms)
        print("✓ Model prediction test successful")
        if result.get('results'):
            top_result = result['results'][0]
            print(f"  Test prediction: {top_result['disease']} ({top_result['probability']}%)")
    except Exception as e:
        print(f"✗ Model prediction test failed: {e}")
        
        print("\n3. Training new model...")
        try:
            accuracy = symptom_checker.train_model()
            print(f"✓ Model trained successfully! Accuracy: {accuracy:.2%}")
        except Exception as e:
            print(f"✗ Model training failed: {e}")
            return False
    
    print("\n4. Final status:")
    print(f"   Model trained: {symptom_checker.is_trained}")
    print(f"   Diseases loaded: {len(symptom_checker.disease_database)}")
    print(f"   Model path: {symptom_checker.model_path}")
    
    print("\n" + "=" * 60)
    print("Initialization completed successfully!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)