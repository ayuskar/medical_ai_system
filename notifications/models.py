from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class Notification(models.Model):
    NOTIFICATION_TYPES = [
        ('appointment_reminder', 'Appointment Reminder'),
        ('appointment_confirmation', 'Appointment Confirmation'),
        ('appointment_cancellation', 'Appointment Cancellation'),
        ('new_message', 'New Message'),
        ('review_reminder', 'Review Reminder'),
        ('system', 'System Notification'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    notification_type = models.CharField(max_length=50, choices=NOTIFICATION_TYPES)
    title = models.CharField(max_length=200)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    related_appointment = models.ForeignKey('appointments.Appointment', on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.notification_type} - {self.user.username}"