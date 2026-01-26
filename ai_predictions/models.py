from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
import json

class SymptomCheckHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='symptom_checks')
    symptoms = models.TextField()
    result_json = models.JSONField(default=dict)
    extracted_info = models.JSONField(default=dict)
    timestamp = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(default=0.0)
    risk_level = models.CharField(max_length=20, default='low')
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['risk_level']),
        ]
    
    def __str__(self):
        return f"Symptom check for {self.user.username} at {self.timestamp}"
    
    def get_top_prediction(self):
        try:
            results = self.result_json.get('results', [])
            if results:
                return results[0]['disease']
            return "No prediction"
        except:
            return "Error"

class DiseaseDatabase(models.Model):
    name = models.CharField(max_length=200, unique=True)
    icd10_code = models.CharField(max_length=20, blank=True)
    symptoms = models.JSONField(default=list)
    severity = models.IntegerField(default=1)  # 1-5 scale
    urgency = models.CharField(max_length=20, default='routine')
    risk_factors = models.JSONField(default=list)
    complications = models.JSONField(default=list)
    treatment_guidelines = models.TextField(blank=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = "Disease Database"

class SymptomPattern(models.Model):
    symptom = models.CharField(max_length=100)
    body_system = models.CharField(max_length=50)
    severity_scale = models.IntegerField(default=1)
    common_associations = models.JSONField(default=list)
    
    def __str__(self):
        return self.symptom

class AIPerformanceLog(models.Model):
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    training_date = models.DateTimeField(auto_now_add=True)
    training_samples = models.IntegerField()
    
    def __str__(self):
        return f"{self.model_name} - {self.accuracy:.2f}%"