from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class SymptomCheckHistory(models.Model):
    """
    Model to store symptom check history for users
    """
    RISK_LEVEL_CHOICES = [
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'),
        ('high', 'High Risk'),
        ('emergency', 'Emergency'),
    ]
    
    URGENCY_CHOICES = [
        ('routine', 'Routine'),
        ('soon', 'Soon'),
        ('urgent', 'Urgent'),
        ('emergency', 'Emergency'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='symptom_checks')
    
    # Symptom information
    symptoms_text = models.TextField()
    symptoms_json = models.JSONField(default=dict, blank=True, null=True)
    
    # Prediction results
    predictions_json = models.JSONField(default=list, blank=True, null=True)
    assessment_json = models.JSONField(default=dict, blank=True, null=True)
    
    # Top prediction
    top_prediction = models.CharField(max_length=200, default=None)
    probability = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        default=0
    )
    
    # Assessment
    risk_level = models.CharField(max_length=20, choices=RISK_LEVEL_CHOICES, default='low')
    urgency = models.CharField(max_length=20, choices=URGENCY_CHOICES, default='routine')
    
    # Metadata
    processing_time = models.FloatField(default=0)  # in seconds
    timestamp = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Additional fields for analysis
    body_system = models.CharField(max_length=100, blank=True, null=True)
    severity_score = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        default=1
    )
    
    # Flags
    is_emergency = models.BooleanField(default=False)
    requires_followup = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['risk_level']),
            models.Index(fields=['top_prediction']),
        ]
        verbose_name_plural = 'Symptom Check Histories'
    
    def __str__(self):
        return f"Symptom check for {self.user.username} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
    
    def get_top_results(self, limit=3):
        """Get top N results from predictions JSON"""
        try:
            if self.predictions_json and isinstance(self.predictions_json, list):
                return self.predictions_json[:limit]
        except:
            pass
        return []
    
    def get_extracted_symptoms(self):
        """Get extracted symptoms from JSON"""
        try:
            if self.symptoms_json and 'extracted_symptoms' in self.symptoms_json:
                return self.symptoms_json['extracted_symptoms']
        except:
            pass
        return []
    
    @property
    def formatted_timestamp(self):
        """Get formatted timestamp"""
        return self.timestamp.strftime('%B %d, %Y at %I:%M %p')
    
    @property
    def days_ago(self):
        """Get days since check"""
        from django.utils import timezone
        delta = timezone.now() - self.timestamp
        return delta.days

class DiseasePattern(models.Model):
    """
    Model to store disease patterns for analysis
    """
    name = models.CharField(max_length=200, unique=True)
    icd10_code = models.CharField(max_length=20, blank=True, null=True)
    symptoms = models.JSONField(default=list)
    severity = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        default=1
    )
    urgency = models.CharField(max_length=20, choices=SymptomCheckHistory.URGENCY_CHOICES, default='routine')
    body_system = models.CharField(max_length=100)
    risk_factors = models.JSONField(default=list, blank=True)
    complications = models.JSONField(default=list, blank=True)
    
    # Statistical data
    prevalence = models.FloatField(default=0)  # Percentage
    avg_age = models.IntegerField(null=True, blank=True)
    gender_distribution = models.JSONField(default=dict, blank=True)  # {'male': 40, 'female': 60}
    
    # ML model data
    ml_confidence = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        default=0.5
    )
    feature_importance = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name

class SymptomPattern(models.Model):
    """
    Model to store symptom patterns
    """
    BODY_SYSTEM_CHOICES = [
        ('respiratory', 'Respiratory'),
        ('cardiovascular', 'Cardiovascular'),
        ('gastrointestinal', 'Gastrointestinal'),
        ('neurological', 'Neurological'),
        ('musculoskeletal', 'Musculoskeletal'),
        ('dermatological', 'Dermatological'),
        ('endocrine', 'Endocrine'),
        ('immune', 'Immune System'),
        ('general', 'General'),
    ]
    
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    body_system = models.CharField(max_length=50, choices=BODY_SYSTEM_CHOICES)
    severity_scale = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        default=1
    )
    common_associations = models.JSONField(default=list, blank=True)
    synonyms = models.JSONField(default=list, blank=True)
    
    # Frequency data
    frequency_score = models.FloatField(default=0)
    seasonal_variation = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        unique_together = ['name', 'body_system']
    
    def __str__(self):
        return f"{self.name} ({self.body_system})"

class AIModelPerformance(models.Model):
    """
    Model to track AI model performance
    """
    model_name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50)
    
    # Performance metrics
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    
    # Training data
    training_samples = models.IntegerField()
    test_samples = models.IntegerField()
    training_date = models.DateTimeField()
    
    # Model parameters
    parameters = models.JSONField(default=dict, blank=True)
    feature_count = models.IntegerField()
    
    # Validation
    cross_val_score = models.FloatField(null=True, blank=True)
    confusion_matrix = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-training_date']
        verbose_name_plural = 'AI Model Performances'
    
    def __str__(self):
        return f"{self.model_name} - {self.accuracy:.2%} - {self.training_date.strftime('%Y-%m-%d')}"

class UserFeedback(models.Model):
    """
    Model to store user feedback on predictions
    """
    SATISFACTION_CHOICES = [
        (1, 'Very Dissatisfied'),
        (2, 'Dissatisfied'),
        (3, 'Neutral'),
        (4, 'Satisfied'),
        (5, 'Very Satisfied'),
    ]
    
    symptom_check = models.ForeignKey(
        SymptomCheckHistory, 
        on_delete=models.CASCADE,
        related_name='feedbacks'
    )
    
    # Feedback
    accuracy_rating = models.IntegerField(choices=SATISFACTION_CHOICES)
    usefulness_rating = models.IntegerField(choices=SATISFACTION_CHOICES)
    ease_of_use_rating = models.IntegerField(choices=SATISFACTION_CHOICES)
    
    comments = models.TextField(blank=True)
    
    # Actual outcome (if known)
    actual_diagnosis = models.CharField(max_length=200, blank=True, null=True)
    doctor_visit = models.BooleanField(default=False)
    diagnosis_match = models.BooleanField(null=True, blank=True)
    
    submitted_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-submitted_at']
    
    def __str__(self):
        return f"Feedback for {self.symptom_check} - Accuracy: {self.accuracy_rating}"
    
    @property
    def overall_score(self):
        """Calculate overall feedback score"""
        return (self.accuracy_rating + self.usefulness_rating + self.ease_of_use_rating) / 3