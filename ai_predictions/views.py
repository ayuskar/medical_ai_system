from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .ml_models.sentiment_analysis import sentiment_analyzer
from django.utils import timezone
from .models import SymptomCheckHistory, DiseaseDatabase
from .ml_models.symptom_checker import advanced_symptom_checker
import json
from datetime import datetime

@login_required
def symptom_checker_view(request):
    """Advanced symptom checker view"""
    results = None
    extracted_info = None
    assessment = None
    
    if request.method == 'POST':
        symptoms_text = request.POST.get('symptoms', '').strip()
        
        if symptoms_text:
            if len(symptoms_text) < 10:
                messages.warning(request, "Please provide more detailed symptoms (at least 10 characters)")
            else:
                try:
                    # Get prediction with explanations
                    prediction_result = advanced_symptom_checker.predict_with_explanations(symptoms_text)
                    
                    results = prediction_result['results']
                    extracted_info = prediction_result['extracted_info']
                    assessment = prediction_result['assessment']
                    
                    # Save to history
                    if request.user.is_authenticated:
                        history_entry = SymptomCheckHistory.objects.create(
                            user=request.user,
                            symptoms=symptoms_text,
                            result_json=json.dumps(prediction_result),
                            extracted_info=json.dumps(extracted_info),
                            timestamp=timezone.now()
                        )
                        history_entry.save()
                    
                    # Add success message
                    messages.success(request, f"Analysis complete! Found {len(results)} possible conditions.")
                    
                except Exception as e:
                    messages.error(request, f"Error analyzing symptoms: {str(e)}")
                    # Fallback to simple analysis
                    results = simple_symptom_check(symptoms_text)
        else:
            messages.error(request, "Please describe your symptoms")
    
    # Get user history
    history = []
    if request.user.is_authenticated:
        history = SymptomCheckHistory.objects.filter(
            user=request.user
        ).order_by('-timestamp')[:5]
    
    # Get trending symptoms/diseases
    trending = get_trending_analysis()
    
    context = {
        'results': results,
        'extracted_info': extracted_info,
        'assessment': assessment,
        'history': history,
        'trending': trending,
        'current_year': datetime.now().year,
    }
    
    return render(request, 'ai_predictions/symptom_checker.html', context)

@login_required
def symptom_history_view(request):
    """View symptom check history"""
    history = SymptomCheckHistory.objects.filter(
        user=request.user
    ).order_by('-timestamp')
    
    return render(request, 'ai_predictions/symptom_history.html', {
        'history': history
    })

@login_required
def symptom_detail_view(request, history_id):
    """View detailed symptom check result"""
    try:
        entry = SymptomCheckHistory.objects.get(id=history_id, user=request.user)
        result_data = json.loads(entry.result_json)
        
        return render(request, 'ai_predictions/symptom_detail.html', {
            'entry': entry,
            'results': result_data.get('results', []),
            'assessment': result_data.get('assessment', {}),
            'extracted_info': result_data.get('extracted_info', {})
        })
    except SymptomCheckHistory.DoesNotExist:
        messages.error(request, "History entry not found")
        return redirect('symptom_checker')

def simple_symptom_check(symptoms_text):
    """Fallback simple symptom checker"""
    # Basic keyword matching
    keywords = {
        'fever': ['Common Cold', 'Influenza', 'COVID-19'],
        'cough': ['Common Cold', 'Influenza', 'COVID-19', 'Bronchitis'],
        'headache': ['Migraine', 'Tension Headache', 'Influenza'],
        'fatigue': ['Influenza', 'COVID-19', 'Anemia']
    }
    
    results = []
    for symptom, diseases in keywords.items():
        if symptom in symptoms_text.lower():
            for disease in diseases[:2]:
                results.append({
                    'disease': disease,
                    'probability': 40.0,
                    'severity': 2
                })
    
    return results[:5]

def get_trending_analysis():
    """Get trending symptoms and diseases"""
    # This could connect to a real analytics database
    return {
        'common_symptoms': ['cough', 'fever', 'headache', 'fatigue'],
        'trending_diseases': ['Influenza', 'Common Cold', 'COVID-19'],
        'seasonal_alert': 'Flu season is active. Consider vaccination.'
    }

@login_required
def analyze_sentiment(request):
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        text = request.POST.get('text')
        if text:
            sentiment = sentiment_analyzer.analyze_sentiment(text)
            return JsonResponse(sentiment)
    
    return JsonResponse({'error': 'Invalid request'})

def prediction_dashboard(request):
    return render(request, 'ai_predictions/dashboard.html')