from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.http import JsonResponse
from django.core.mail import send_mail
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.utils import timezone
from datetime import timedelta
import logging

from .models import ContactMessage, ContactCategory, FAQ, SupportTicket
from .forms import ContactForm, ContactResponseForm, FAQForm

logger = logging.getLogger(__name__)

def contact_page(request):
    """Main contact page with form and information"""
    faqs = FAQ.objects.filter(is_active=True).order_by('category', 'order')
    categories = ContactCategory.objects.filter(is_active=True)
    
    if request.method == 'POST':
        form = ContactForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            # Validate email
            email = form.cleaned_data['email']
            try:
                validate_email(email)
            except ValidationError:
                messages.error(request, 'Please enter a valid email address.')
                return render(request, 'contact/contact_page.html', {
                    'form': form,
                    'faqs_by_category': faqs_by_category,
                    'categories': categories,
                })
            
            # Check for recent submissions (prevent spam)
            recent_submission = ContactMessage.objects.filter(
                email=email,
                created_at__gte=timezone.now() - timedelta(hours=1)
            ).exists()
            
            if recent_submission:
                messages.error(request, 'You have already submitted a message recently. Please wait 1 hour before submitting again.')
                return render(request, 'contact/contact_page.html', {
                    'form': form,
                    'faqs_by_category': faqs_by_category,
                    'categories': categories,
                })
            
            # Save contact message
            contact_message = form.save(commit=False)
            if request.user.is_authenticated:
                contact_message.user = request.user
            contact_message.save()
            
            # Send email notifications
            send_contact_emails(contact_message, request)
            
            messages.success(
                request, 
                'Thank you for your message! We have received your inquiry and will get back to you within 24 hours. Check your email for confirmation!'
            )
            return redirect('contact:contact_success')
    else:
        form = ContactForm(user=request.user)
    
    # Group FAQs by category
    faqs_by_category = {}
    for faq in faqs:
        if faq.category not in faqs_by_category:
            faqs_by_category[faq.category] = []
        faqs_by_category[faq.category].append(faq)
    
    context = {
        'form': form,
        'faqs_by_category': faqs_by_category,
        'categories': categories,
    }
    return render(request, 'contact/contact_page.html', context)

def send_contact_emails(contact_message, request):
    """Send email notifications to user and admin"""
    try:
        # 1. Thank you email to user
        user_subject = "Thank You for Contacting MediCare AI"
        user_message = f"""
Dear {contact_message.name},

Thank you for reaching out to MediCare AI! We have received your message and appreciate you taking the time to contact us.

Here's a summary of your inquiry:
- Category: {contact_message.category.get_name_display() if contact_message.category else 'General'}
- Subject: {contact_message.subject}
- Message: {contact_message.message}

Our support team will review your message and get back to you within 24 hours. We strive to respond to all inquiries as quickly as possible.

If you have any urgent matters, please feel free to call us directly at +1 (555) 123-4567.

You can also view the status of your inquiry by logging into your account and visiting the "My Messages" section.

Best regards,
The MediCare AI Support Team

MediCare AI - Smart Healthcare Solutions
Email: support@medicareai.com
Phone: +1 (555) 123-4567
"""

        send_mail(
            user_subject,
            user_message.strip(),
            settings.DEFAULT_FROM_EMAIL,
            [contact_message.email],
            fail_silently=False,
        )
        
        # 2. Notification email to admin
        admin_subject = f"📧 New Contact Form: {contact_message.subject}"
        admin_message = f"""
New contact form submission received from MediCare AI website:

CONTACT DETAILS:
───────────────
• Name: {contact_message.name}
• Email: {contact_message.email}
• Phone: {contact_message.phone or 'Not provided'}
• User Account: {contact_message.user.username if contact_message.user else 'Not logged in'}

INQUIRY DETAILS:
───────────────
• Category: {contact_message.category.get_name_display() if contact_message.category else 'General'}
• Subject: {contact_message.subject}
• Priority: {contact_message.get_priority_display()}
• Status: {contact_message.get_status_display()}

MESSAGE:
────────
{contact_message.message}

ADDITIONAL INFO:
────────────────
• Submitted: {contact_message.created_at.strftime('%Y-%m-%d %H:%M:%S')}
• Message ID: #{contact_message.id}
• Attachment: {'Yes' if contact_message.attachment else 'No'}

Please respond within 24 hours by:
1. Logging into the admin panel
2. Finding this message (ID: {contact_message.id})
3. Sending a professional response

Urgency: {'🚨 URGENT - Please respond immediately' if contact_message.is_urgent else 'Normal priority'}
"""

        # Send to admin email from settings
        admin_emails = [settings.ADMIN_EMAIL] if hasattr(settings, 'ADMIN_EMAIL') else [settings.EMAIL_HOST_USER]
        
        send_mail(
            admin_subject,
            admin_message.strip(),
            settings.DEFAULT_FROM_EMAIL,
            admin_emails,
            fail_silently=False,
        )
        
        logger.info(f"Contact emails sent successfully for message ID: {contact_message.id}")
        
    except Exception as e:
        logger.error(f"Failed to send contact emails: {str(e)}")
        # Don't show error to user, just log it
        pass

def contact_success(request):
    """Success page after submitting contact form"""
    return render(request, 'contact/contact_success.html')

def faq_list(request):
    """Dedicated FAQ page"""
    faqs = FAQ.objects.filter(is_active=True).order_by('category', 'order')
    
    # Group FAQs by category
    faqs_by_category = {}
    for faq in faqs:
        if faq.category not in faqs_by_category:
            faqs_by_category[faq.category] = []
        faqs_by_category[faq.category].append(faq)
    
    context = {
        'faqs_by_category': faqs_by_category,
    }
    return render(request, 'contact/faq_list.html', context)

@login_required
def my_contact_messages(request):
    """User's previous contact messages"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    messages_list = ContactMessage.objects.filter(
        Q(user=request.user) | Q(email=request.user.email)
    ).order_by('-created_at')
    
    paginator = Paginator(messages_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_messages': messages_list.count(),
    }
    return render(request, 'contact/my_contact_messages.html', context)

# Admin views
def is_staff_user(user):
    return user.is_staff

@user_passes_test(is_staff_user)
@login_required
def contact_admin(request):
    """Admin panel for managing contact messages"""
    messages_list = ContactMessage.objects.all().order_by('-created_at')
    
    # Filters
    status_filter = request.GET.get('status')
    priority_filter = request.GET.get('priority')
    category_filter = request.GET.get('category')
    
    if status_filter:
        messages_list = messages_list.filter(status=status_filter)
    if priority_filter:
        messages_list = messages_list.filter(priority=priority_filter)
    if category_filter:
        messages_list = messages_list.filter(category__name=category_filter)
    
    # Statistics
    total_messages = messages_list.count()
    new_messages = messages_list.filter(status='new').count()
    urgent_messages = messages_list.filter(priority='urgent', status__in=['new', 'in_progress']).count()
    
    paginator = Paginator(messages_list, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_messages': total_messages,
        'new_messages': new_messages,
        'urgent_messages': urgent_messages,
        'categories': ContactCategory.objects.filter(is_active=True),
    }
    return render(request, 'contact/contact_admin.html', context)

@user_passes_test(is_staff_user)
@login_required
def contact_message_detail(request, message_id):
    """Admin view for individual contact message"""
    contact_message = get_object_or_404(ContactMessage, id=message_id)
    
    if request.method == 'POST':
        form = ContactResponseForm(request.POST)
        if form.is_valid():
            response_message = form.cleaned_data['response_message']
            mark_as_resolved = form.cleaned_data['mark_as_resolved']
            
            try:
                # Send response email to user
                send_admin_response_email(contact_message, response_message, request.user)
                
                # Update contact message
                contact_message.response_message = response_message
                contact_message.response_sent = True
                contact_message.response_date = timezone.now()
                contact_message.responded_by = request.user
                
                if mark_as_resolved:
                    contact_message.status = 'resolved'
                    contact_message.resolved_at = timezone.now()
                
                contact_message.save()
                
                messages.success(request, 'Response sent successfully to the user!')
                
            except Exception as e:
                logger.error(f"Failed to send admin response email: {str(e)}")
                messages.warning(request, 'Response was saved but there was an issue sending the email.')
            
            return redirect('contact:contact_admin')
    else:
        form = ContactResponseForm()
    
    context = {
        'contact_message': contact_message,
        'form': form,
    }
    return render(request, 'contact/contact_message_detail.html', context)

def send_admin_response_email(contact_message, response_text, admin_user):
    """Send response email from admin to user"""
    try:
        subject = f"Re: {contact_message.subject}"
        message = f"""
Dear {contact_message.name},

Thank you for contacting MediCare AI. Here is our response to your inquiry:

{response_text}

Original Message:
────────────────
{contact_message.message}

If you have any further questions or need additional assistance, please don't hesitate to contact us again.

Best regards,
{admin_user.get_full_name() or admin_user.username}
MediCare AI Support Team

MediCare AI - Smart Healthcare Solutions
Email: support@medicareai.com
Phone: +1 (555) 123-4567

Note: This is an automated response. Please do not reply directly to this email.
"""

        send_mail(
            subject,
            message.strip(),
            settings.DEFAULT_FROM_EMAIL,
            [contact_message.email],
            fail_silently=False,
        )
        
        logger.info(f"Admin response email sent for message ID: {contact_message.id}")
        
    except Exception as e:
        logger.error(f"Failed to send admin response email: {str(e)}")
        raise

@user_passes_test(is_staff_user)
@login_required
def update_message_status(request, message_id):
    """Update message status via AJAX"""
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        contact_message = get_object_or_404(ContactMessage, id=message_id)
        new_status = request.POST.get('status')
        
        if new_status in dict(ContactMessage.STATUS_CHOICES):
            contact_message.status = new_status
            if new_status == 'resolved':
                contact_message.resolved_at = timezone.now()
            contact_message.save()
            
            return JsonResponse({
                'success': True,
                'new_status': contact_message.get_status_display()
            })
    
    return JsonResponse({'success': False})

# API views for AJAX
def get_faq_by_category(request):
    """Get FAQs by category for AJAX requests"""
    category = request.GET.get('category')
    if category:
        faqs = FAQ.objects.filter(category=category, is_active=True).order_by('order')
        data = [{'question': faq.question, 'answer': faq.answer} for faq in faqs]
        return JsonResponse(data, safe=False)
    return JsonResponse([], safe=False)