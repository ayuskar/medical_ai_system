from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .ml_models.symptom_checker import symptom_checker
from .ml_models.sentiment_analysis import sentiment_analyzer

@login_required
def symptom_checker_view(request):
    results = None
    
    if request.method == 'POST':
        symptoms = request.POST.get('symptoms')
        if symptoms:
            results = symptom_checker.predict(symptoms)
    
    return render(request, 'ai_predictions/symptom_checker.html', {'results': results})

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