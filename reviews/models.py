from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
from appointments.models import Appointment

class Review(models.Model):
    appointment = models.OneToOneField(Appointment, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=[(i, f'{i} star') for i in range(1, 6)])
    review_text = models.TextField()
    sentiment = models.CharField(max_length=20, blank=True)
    sentiment_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Review for {self.appointment} - {self.rating} stars"
    
    def save(self, *args, **kwargs):
        # Auto-analyze sentiment when review is saved
        from ai_predictions.ml_models.sentiment_analysis import sentiment_analyzer
        if self.review_text and (not self.sentiment or not self.sentiment_score):
            sentiment_result = sentiment_analyzer.analyze_sentiment(self.review_text)
            self.sentiment = sentiment_result['label']
            self.sentiment_score = sentiment_result['score']
        super().save(*args, **kwargs)