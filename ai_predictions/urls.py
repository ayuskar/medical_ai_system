from django.urls import path
from . import views

app_name = 'ai_predictions'

urlpatterns = [
    # Symptom Checker
    path('symptom-checker/', views.symptom_checker_view, name='symptom_checker'),
    path('symptom-history/', views.symptom_history_view, name='symptom_history'),
    path('symptom-detail/<int:check_id>/', views.symptom_detail_view, name='symptom_detail'),
    path('delete-check/<int:check_id>/', views.delete_symptom_check, name='delete_check'),
    
    # API Endpoints
    path('api/quick-check/', views.quick_check_api, name='quick_check_api'),
    
    # Model Management
    path('model-info/', views.model_info_view, name='model_info'),
    path('retrain-model/', views.retrain_model_view, name='retrain_model'),
    
    # Emergency
    path('emergency-check/', views.emergency_check, name='emergency_check'),
]