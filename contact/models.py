from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class ContactCategory(models.Model):
    CATEGORY_CHOICES = [
        ('general', 'General Inquiry'),
        ('technical', 'Technical Support'),
        ('billing', 'Billing Issue'),
        ('medical', 'Medical Question'),
        ('feedback', 'Feedback/Suggestion'),
        ('complaint', 'Complaint'),
        ('partnership', 'Partnership'),
        ('other', 'Other'),
    ]
    
    name = models.CharField(max_length=50, choices=CATEGORY_CHOICES, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.get_name_display()

class ContactMessage(models.Model):
    STATUS_CHOICES = [
        ('new', 'New'),
        ('in_progress', 'In Progress'),
        ('resolved', 'Resolved'),
        ('closed', 'Closed'),
    ]
    
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]
    
    # Sender information
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='contact_messages')
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=20, blank=True)
    
    # Message details
    category = models.ForeignKey(ContactCategory, on_delete=models.SET_NULL, null=True)
    subject = models.CharField(max_length=200)
    message = models.TextField()
    attachment = models.FileField(upload_to='contact_attachments/', blank=True, null=True)
    
    # Status and tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='medium')
    assigned_to = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='assigned_contacts')
    
    # Response tracking
    admin_notes = models.TextField(blank=True)
    response_sent = models.BooleanField(default=False)
    response_message = models.TextField(blank=True)
    responded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='responded_contacts')
    response_date = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Contact Message'
        verbose_name_plural = 'Contact Messages'
    
    def __str__(self):
        return f"{self.subject} - {self.name}"
    
    def mark_as_resolved(self, user=None):
        self.status = 'resolved'
        self.resolved_at = timezone.now()
        if user:
            self.responded_by = user
        self.save()
    
    def send_response(self, response_message, user=None):
        self.response_message = response_message
        self.response_sent = True
        self.response_date = timezone.now()
        if user:
            self.responded_by = user
        self.save()
        
        # Here you would integrate with your email service
        # to actually send the response email
        self._send_response_email()
    
    def _send_response_email(self):
        """Send response email to the contact message sender"""
        # This is a placeholder for email integration
        # You would integrate with Django's email backend or a service like SendGrid
        try:
            # Example email sending (you need to configure email settings)
            from django.core.mail import send_mail
            from django.conf import settings
            
            subject = f"Re: {self.subject}"
            message = f"""
            Dear {self.name},
            
            Thank you for contacting MediCare AI. Here is our response to your inquiry:
            
            {self.response_message}
            
            If you have any further questions, please don't hesitate to contact us again.
            
            Best regards,
            MediCare AI Support Team
            """
            
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[self.email],
                fail_silently=True,
            )
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    @property
    def is_urgent(self):
        return self.priority == 'urgent'
    
    @property
    def days_open(self):
        if self.resolved_at:
            return (self.resolved_at - self.created_at).days
        return (timezone.now() - self.created_at).days

class FAQ(models.Model):
    CATEGORY_CHOICES = [
        ('general', 'General'),
        ('technical', 'Technical'),
        ('billing', 'Billing'),
        ('medical', 'Medical'),
        ('privacy', 'Privacy & Security'),
        ('account', 'Account'),
    ]
    
    question = models.CharField(max_length=255)
    answer = models.TextField()
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='general')
    order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['category', 'order', 'question']
        verbose_name = 'FAQ'
        verbose_name_plural = 'FAQs'
    
    def __str__(self):
        return self.question

class SupportTicket(models.Model):
    """Extended support system for complex issues"""
    contact_message = models.OneToOneField(ContactMessage, on_delete=models.CASCADE, related_name='support_ticket')
    ticket_id = models.CharField(max_length=20, unique=True)
    department = models.CharField(max_length=50, choices=[
        ('technical', 'Technical Support'),
        ('medical', 'Medical Team'),
        ('billing', 'Billing Department'),
        ('admin', 'Administration'),
    ])
    severity = models.CharField(max_length=20, choices=[
        ('low', 'Low - General Inquiry'),
        ('medium', 'Medium - Feature Request'),
        ('high', 'High - Bug Report'),
        ('critical', 'Critical - System Down'),
    ])
    estimated_resolution_time = models.DurationField(null=True, blank=True)
    
    def save(self, *args, **kwargs):
        if not self.ticket_id:
            self.ticket_id = self.generate_ticket_id()
        super().save(*args, **kwargs)
    
    def generate_ticket_id(self):
        import uuid
        return f"TKT-{uuid.uuid4().hex[:8].upper()}"
    
    def __str__(self):
        return f"{self.ticket_id} - {self.contact_message.subject}"