from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class Doctor(models.Model):
    SPECIALIZATION_CHOICES = [
        ('cardiology', 'Cardiology'),
        ('neurology', 'Neurology'),
        ('pediatrics', 'Pediatrics'),
        ('orthopedics', 'Orthopedics'),
        ('dermatology', 'Dermatology'),
        ('psychiatry', 'Psychiatry'),
        ('general', 'General Medicine'),
        ('dentistry', 'Dentistry'),
        ('ophthalmology', 'Ophthalmology'),
        ('surgery', 'Surgery'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    specialization = models.CharField(max_length=50, choices=SPECIALIZATION_CHOICES)
    experience = models.IntegerField(default=0)
    about = models.TextField(blank=True)
    consultation_fee = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    license_number = models.CharField(max_length=50, blank=True)
    is_available = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Dr. {self.user.get_full_name()} - {self.specialization}"
    
    def get_rating(self):
        from reviews.models import Review
        from django.db.models import Avg
        reviews = Review.objects.filter(appointment__doctor=self.user)
        if reviews.exists():
            return reviews.aggregate(Avg('rating'))['rating__avg']
        return 0
    
    def get_review_count(self):
        from reviews.models import Review
        return Review.objects.filter(appointment__doctor=self.user).count()

class Availability(models.Model):
    DAY_CHOICES = [
        ('monday', 'Monday'),
        ('tuesday', 'Tuesday'),
        ('wednesday', 'Wednesday'),
        ('thursday', 'Thursday'),
        ('friday', 'Friday'),
        ('saturday', 'Saturday'),
        ('sunday', 'Sunday'),
    ]
    
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='availabilities')
    day = models.CharField(max_length=10, choices=DAY_CHOICES)
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_available = models.BooleanField(default=True)

    class Meta:
        unique_together = ['doctor', 'day']
        verbose_name_plural = 'Availabilities'

    def __str__(self):
        return f"{self.doctor.user.get_full_name()} - {self.day}"