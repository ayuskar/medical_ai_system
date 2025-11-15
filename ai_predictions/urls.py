from django.urls import path
from . import views

app_name = 'ai_predictions'

urlpatterns = [
    path('symptom-checker/', views.symptom_checker_view, name='symptom_checker'),
    path('analyze-sentiment/', views.analyze_sentiment, name='analyze_sentiment'),
    path('dashboard/', views.prediction_dashboard, name='prediction_dashboard'),
]