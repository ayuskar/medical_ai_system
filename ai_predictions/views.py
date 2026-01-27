from django.shortcuts import render

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from datetime import datetime, timedelta

from .models import SymptomCheckHistory
from .ml_models.symptom_checker import symptom_checker
from .ml_models.sentiment_analysis import sentiment_analyzer
@login_required
def symptom_checker_view(request):
    """
    Main symptom checker view
    """
    results = None
    assessment = None
    extracted_info = None
    processing_time = None
    
    if request.method == 'POST':
        symptoms_text = request.POST.get('symptoms', '').strip()
        
        if not symptoms_text:
            messages.error(request, "Please describe your symptoms")
        elif len(symptoms_text) < 10:
            messages.warning(request, "Please provide more detailed symptoms (at least 10 characters)")
        else:
            try:
                # Get analysis from symptom checker
                analysis_result = symptom_checker.analyze_symptoms(symptoms_text)
                
                results = analysis_result.get('results', [])
                assessment = analysis_result.get('assessment', {})
                extracted_info = analysis_result.get('extracted_info', {})
                processing_time = analysis_result.get('processing_time', 0)
                model_info = analysis_result.get('model_info', {})
                
                # Save to history if user is authenticated
                if request.user.is_authenticated and results:
                    try:
                        history_entry = SymptomCheckHistory.objects.create(
                            user=request.user,
                            symptoms_text=symptoms_text,
                            symptoms_json=json.dumps(extracted_info),
                            predictions_json=json.dumps(results),
                            assessment_json=json.dumps(assessment),
                            top_prediction=results[0]['disease'] if results else 'Unknown',
                            probability=results[0]['probability'] if results else 0,
                            risk_level=assessment.get('risk_level', 'low'),
                            processing_time=processing_time,
                            timestamp=timezone.now()
                        )
                        history_entry.save()
                        
                        # Update session for recent checks
                        recent_checks = request.session.get('recent_symptom_checks', [])
                        recent_checks.insert(0, {
                            'id': history_entry.id,
                            'symptoms': symptoms_text[:50] + '...' if len(symptoms_text) > 50 else symptoms_text,
                            'prediction': results[0]['disease'] if results else 'Unknown',
                            'timestamp': timezone.now().isoformat()
                        })
                        # Keep only last 5 checks
                        request.session['recent_symptom_checks'] = recent_checks[:5]
                        request.session.modified = True
                        
                    except Exception as e:
                        print(f"Error saving to history: {e}")
                        # Continue even if history save fails
                
                # Prepare success message
                if results:
                    top_result = results[0]
                    messages.success(
                        request, 
                        f"Analysis complete! Top match: {top_result['disease']} "
                        f"({top_result['probability']}% probability)"
                    )
                else:
                    messages.info(request, "Analysis complete. No specific conditions identified.")
                
                # Add model info to context
                request.session['last_model_info'] = model_info
                
            except Exception as e:
                print(f"Error in symptom analysis: {e}")
                messages.error(
                    request, 
                    f"Error analyzing symptoms: {str(e)}. Please try again with different wording."
                )
                # Provide fallback options
                results = [{
                    'disease': 'Analysis Error',
                    'probability': 0,
                    'recommendation': 'Please try describing your symptoms differently or consult a healthcare professional.',
                    'severity': 1,
                    'urgency': 'routine'
                }]
    
    # Get user's recent checks
    recent_history = []
    if request.user.is_authenticated:
        recent_history = SymptomCheckHistory.objects.filter(
            user=request.user
        ).order_by('-timestamp')[:5]
    
    # Get common symptoms for suggestions
    common_symptoms = [
        {'name': 'Fever', 'icon': 'fas fa-thermometer-half', 'color': 'text-red-500'},
        {'name': 'Cough', 'icon': 'fas fa-lungs', 'color': 'text-blue-500'},
        {'name': 'Headache', 'icon': 'fas fa-head-side-virus', 'color': 'text-purple-500'},
        {'name': 'Fatigue', 'icon': 'fas fa-tired', 'color': 'text-yellow-500'},
        {'name': 'Nausea', 'icon': 'fas fa-stomach', 'color': 'text-green-500'},
        {'name': 'Body Aches', 'icon': 'fas fa-pain', 'color': 'text-orange-500'},
    ]
    
    # Get session model info
    model_info = request.session.get('last_model_info', {})
    
    context = {
        'results': results,
        'assessment': assessment,
        'extracted_info': extracted_info,
        'processing_time': processing_time,
        'recent_history': recent_history,
        'common_symptoms': common_symptoms,
        'model_info': model_info,
        'current_year': datetime.now().year,
        'total_diseases': len(symptom_checker.disease_database) if hasattr(symptom_checker, 'disease_database') else 0,
    }
    
    return render(request, 'ai_predictions/symptom_checker.html', context)

@login_required
def symptom_history_view(request):
    """
    View symptom check history
    """
    history_list = SymptomCheckHistory.objects.filter(
        user=request.user
    ).order_by('-timestamp')
    
    # Pagination
    paginator = Paginator(history_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistics
    total_checks = history_list.count()
    if total_checks > 0:
        avg_probability = history_list.aggregate(models.Avg('probability'))['probability__avg']
        common_prediction = history_list.values('top_prediction').annotate(
            count=models.Count('id')
        ).order_by('-count').first()
    else:
        avg_probability = 0
        common_prediction = None
    
    context = {
        'page_obj': page_obj,
        'total_checks': total_checks,
        'avg_probability': round(avg_probability, 1) if avg_probability else 0,
        'common_prediction': common_prediction,
    }
    
    return render(request, 'ai_predictions/symptom_history.html', context)

@login_required
def symptom_detail_view(request, check_id):
    """
    View detailed symptom check result
    """
    try:
        check = SymptomCheckHistory.objects.get(id=check_id, user=request.user)
        
        # Parse JSON data
        symptoms_data = json.loads(check.symptoms_json) if check.symptoms_json else {}
        predictions_data = json.loads(check.predictions_json) if check.predictions_json else []
        assessment_data = json.loads(check.assessment_json) if check.assessment_json else {}
        
        context = {
            'check': check,
            'symptoms_data': symptoms_data,
            'predictions_data': predictions_data,
            'assessment_data': assessment_data,
        }
        
        return render(request, 'ai_predictions/symptom_detail.html', context)
        
    except SymptomCheckHistory.DoesNotExist:
        messages.error(request, "Symptom check not found")
        return redirect('ai_predictions:symptom_checker')

@login_required
@require_POST
def delete_symptom_check(request, check_id):
    """
    Delete a symptom check from history
    """
    try:
        check = SymptomCheckHistory.objects.get(id=check_id, user=request.user)
        check.delete()
        messages.success(request, "Symptom check deleted successfully")
    except SymptomCheckHistory.DoesNotExist:
        messages.error(request, "Symptom check not found")
    
    return redirect('ai_predictions:symptom_history')

def quick_check_api(request):
    """
    API endpoint for quick symptom check (AJAX)
    """
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        symptoms_text = request.POST.get('symptoms', '').strip()
        
        if not symptoms_text or len(symptoms_text) < 5:
            return JsonResponse({
                'success': False,
                'error': 'Please provide at least 5 characters of symptoms'
            })
        
        try:
            # Quick analysis (limit to top 3)
            analysis_result = symptom_checker.analyze_symptoms(symptoms_text)
            results = analysis_result.get('results', [])[:3]
            assessment = analysis_result.get('assessment', {})
            
            return JsonResponse({
                'success': True,
                'results': results,
                'assessment': assessment,
                'processing_time': analysis_result.get('processing_time', 0)
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

@login_required
def retrain_model_view(request):
    """
    View to retrain the symptom checker model (admin/advanced users)
    """
    if not request.user.is_superuser:
        messages.error(request, "You don't have permission to retrain the model")
        return redirect('ai_predictions:symptom_checker')
    
    if request.method == 'POST':
        try:
            accuracy = symptom_checker.train_model()
            messages.success(
                request, 
                f"Model retrained successfully! Accuracy: {accuracy:.2%}"
            )
        except Exception as e:
            messages.error(request, f"Error retraining model: {str(e)}")
    
    return render(request, 'ai_predictions/retrain_model.html')

def emergency_check(request):
    """
    Emergency symptom check for critical symptoms
    """
    critical_symptoms = [
        'chest pain', 'difficulty breathing', 'severe bleeding',
        'sudden weakness', 'loss of consciousness', 'severe headache'
    ]
    
    context = {
        'critical_symptoms': critical_symptoms,
        'emergency_number': '999',  # UK emergency number
    }
    
    return render(request, 'ai_predictions/emergency_check.html', context)

def model_info_view(request):
    """
    View model information and statistics
    """
    model_info = {
        'type': 'Ensemble Machine Learning Model',
        'components': [
            'Random Forest Classifier',
            'Gradient Boosting Classifier', 
            'Multinomial Naive Bayes'
        ],
        'diseases_covered': len(symptom_checker.disease_database),
        'training_samples': 150 * len(symptom_checker.disease_database),  # Approximate
        'accuracy': '85-90% (estimated)',
        'last_trained': 'On application start',
        'features': 'TF-IDF with n-grams (1-2)',
        'vectorizer_features': 2000,
    }
    
    # Get performance stats from session
    performance_stats = request.session.get('model_performance', {})
    
    context = {
        'model_info': model_info,
        'performance_stats': performance_stats,
        'is_trained': symptom_checker.is_trained,
    }
    
    return render(request, 'ai_predictions/model_info.html', context)

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