from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Appointment(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('no_show', 'No Show'),
    ]
    
    patient = models.ForeignKey(User, related_name='patient_appointments', on_delete=models.CASCADE)
    doctor = models.ForeignKey(User, related_name='doctor_appointments', on_delete=models.CASCADE)
    date = models.DateField()
    time = models.TimeField()
    reason = models.TextField()
    symptoms = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    no_show = models.BooleanField(default=False)
    prediction_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date', '-time']

    def __str__(self):
        return f"Appointment: {self.patient.username} with Dr. {self.doctor.username} on {self.date}"
    
    def is_upcoming(self):
        return self.date >= timezone.now().date() and self.status in ['pending', 'confirmed']
    
    def can_be_cancelled(self):
        # Allow cancellation if appointment is pending/confirmed and date is in the future
        from django.utils import timezone
        from datetime import datetime
        appointment_datetime = datetime.combine(self.date, self.time)
        current_datetime = timezone.now()
        return self.status in ['pending', 'confirmed'] and appointment_datetime > current_datetime

class TimeSlot(models.Model):
    doctor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='time_slots')
    date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_available = models.BooleanField(default=True)
    appointment = models.OneToOneField(Appointment, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.doctor.username} - {self.date} {self.start_time}"