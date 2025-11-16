from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
import logging

from .models import Appointment, TimeSlot
from .forms import AppointmentForm, TimeSlotForm
from doctors.models import Doctor
from ai_predictions.ml_models.no_show_model import no_show_predictor

logger = logging.getLogger(__name__)

@login_required
def appointments_list(request):
    user = request.user
    
    if user.profile.role == 'patient':
        appointments = Appointment.objects.filter(patient=user)
    else:
        appointments = Appointment.objects.filter(doctor=user)
    
    # Filter by status
    status_filter = request.GET.get('status')
    if status_filter:
        appointments = appointments.filter(status=status_filter)
    
    # Define appointment statuses for the template
    appointment_statuses = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('no_show', 'No Show'),
    ]
    
    # Calculate stats
    total_appointments = appointments.count()
    upcoming_appointments = appointments.filter(
        date__gte=timezone.now().date(),
        status__in=['pending', 'confirmed']
    ).count()
    pending_appointments = appointments.filter(status='pending').count()
    completed_appointments = appointments.filter(status='completed').count()
    
    context = {
        'appointments': appointments.order_by('-date', '-time'),
        'appointment_statuses': appointment_statuses,
        'total_appointments': total_appointments,
        'upcoming_appointments': upcoming_appointments,
        'pending_appointments': pending_appointments,
        'completed_appointments': completed_appointments,
    }
    return render(request, 'appointments/appointments_list.html', context)

@login_required
def update_appointment_status(request, appointment_id):
    appointment = get_object_or_404(Appointment, id=appointment_id)
    
    # Check permission
    if request.user != appointment.doctor and request.user != appointment.patient:
        messages.error(request, 'You do not have permission to update this appointment.')
        return redirect('appointments:appointments_list')
    
    if request.method == 'POST':
        old_status = appointment.status
        new_status = request.POST.get('status')
        
        if new_status in dict(Appointment.STATUS_CHOICES):
            appointment.status = new_status
            appointment.save()
            
            # Send status update emails
            send_appointment_status_update_emails(appointment, old_status, new_status, request.user)
            
            messages.success(request, f'Appointment status updated to {new_status}.')
    
    return redirect('appointments:appointments_list')

@login_required
def appointment_detail(request, appointment_id):
    appointment = get_object_or_404(Appointment, id=appointment_id)
    
    # Check permission
    if request.user != appointment.doctor and request.user != appointment.patient:
        messages.error(request, 'You do not have permission to view this appointment.')
        return redirect('appointments:appointments_list')
    
    context = {
        'appointment': appointment,
    }
    return render(request, 'appointments/appointment_detail.html', context)

@login_required
def book_appointment(request, doctor_id):
    from django.contrib.auth.models import User
    doctor_user = get_object_or_404(User, doctor__id=doctor_id, doctor__is_available=True)
    doctor = doctor_user.doctor
    
    if request.method == 'POST':
        form = AppointmentForm(request.POST)
        if form.is_valid():
            appointment = form.save(commit=False)
            appointment.patient = request.user
            appointment.doctor = doctor_user
            
            # Check for existing appointment at same time
            existing_appointment = Appointment.objects.filter(
                doctor=doctor_user,
                date=appointment.date,
                time=appointment.time,
                status__in=['pending', 'confirmed']
            ).exists()
            
            if existing_appointment:
                messages.error(request, 'This time slot is already booked. Please choose another time.')
            else:
                appointment.save()
                
                # Predict no-show probability
                features = {
                    'age': request.user.profile.get_age() or 40,
                    'gender': 'Male' if request.user.first_name else 'Female',
                    'day_of_week': appointment.date.strftime('%A'),
                    'lead_time': (appointment.date - timezone.now().date()).days,
                    'previous_no_shows': Appointment.objects.filter(patient=request.user, no_show=True).count(),
                    'sms_sent': 1,
                    'medical_specialty': doctor.specialization
                }
                
                appointment.prediction_score = no_show_predictor.predict(features)
                appointment.save()
                
                # Send appointment confirmation emails
                send_appointment_confirmation_emails(appointment)
                
                messages.success(request, 'Appointment booked successfully! Check your email for confirmation.')
                return redirect('appointments:appointments_list')
    else:
        form = AppointmentForm()
    
    context = {
        'form': form,
        'doctor': doctor,
    }
    return render(request, 'appointments/book_appointment.html', context)

def send_appointment_confirmation_emails(appointment):
    """Send confirmation emails to patient, doctor, and admin when appointment is booked"""
    try:
        # Email data for templates
        email_context = {
            'appointment': appointment,
            'patient': appointment.patient,
            'doctor': appointment.doctor,
            'doctor_profile': appointment.doctor.doctor,
            'appointment_date': appointment.date.strftime('%B %d, %Y'),
            'appointment_time': appointment.time.strftime('%I:%M %p'),
            'no_show_risk': f"{appointment.prediction_score * 100:.1f}%" if appointment.prediction_score else "Not calculated",
        }
        
        # 1. Email to Patient
        patient_subject = f"✅ Appointment Confirmed with Dr. {appointment.doctor.get_full_name()}"
        patient_html_message = render_to_string('appointments/emails/patient_confirmation.html', email_context)
        patient_plain_message = strip_tags(patient_html_message)
        
        send_mail(
            patient_subject,
            patient_plain_message,
            settings.DEFAULT_FROM_EMAIL,
            [appointment.patient.email],
            html_message=patient_html_message,
            fail_silently=False,
        )
        
        # 2. Email to Doctor
        doctor_subject = f"📅 New Appointment Booking - {appointment.patient.get_full_name()}"
        doctor_html_message = render_to_string('appointments/emails/doctor_notification.html', email_context)
        doctor_plain_message = strip_tags(doctor_html_message)
        
        send_mail(
            doctor_subject,
            doctor_plain_message,
            settings.DEFAULT_FROM_EMAIL,
            [appointment.doctor.email],
            html_message=doctor_html_message,
            fail_silently=False,
        )
        
        # 3. Email to Admin
        admin_subject = f"📋 New Appointment Booked - {appointment.patient.get_full_name()} with Dr. {appointment.doctor.get_full_name()}"
        admin_html_message = render_to_string('appointments/emails/admin_notification.html', email_context)
        admin_plain_message = strip_tags(admin_html_message)
        
        admin_emails = [settings.ADMIN_EMAIL] if hasattr(settings, 'ADMIN_EMAIL') else [settings.EMAIL_HOST_USER]
        
        send_mail(
            admin_subject,
            admin_plain_message,
            settings.DEFAULT_FROM_EMAIL,
            admin_emails,
            html_message=admin_html_message,
            fail_silently=False,
        )
        
        logger.info(f"Appointment confirmation emails sent successfully for appointment ID: {appointment.id}")
        
    except Exception as e:
        logger.error(f"Failed to send appointment confirmation emails: {str(e)}")

def send_appointment_status_update_emails(appointment, old_status, new_status, updated_by):
    """Send emails when appointment status is updated"""
    try:
        email_context = {
            'appointment': appointment,
            'patient': appointment.patient,
            'doctor': appointment.doctor,
            'doctor_profile': appointment.doctor.doctor,
            'old_status': old_status,
            'new_status': new_status,
            'updated_by': updated_by,
            'appointment_date': appointment.date.strftime('%B %d, %Y'),
            'appointment_time': appointment.time.strftime('%I:%M %p'),
        }
        
        status_display = dict(Appointment.STATUS_CHOICES).get(new_status, new_status)
        
        if new_status == 'confirmed':
            # Send confirmation to patient
            patient_subject = f"✅ Appointment Confirmed - {appointment_date}"
            patient_html_message = render_to_string('appointments/emails/status_confirmed.html', email_context)
            patient_plain_message = strip_tags(patient_html_message)
            
            send_mail(
                patient_subject,
                patient_plain_message,
                settings.DEFAULT_FROM_EMAIL,
                [appointment.patient.email],
                html_message=patient_html_message,
                fail_silently=False,
            )
            
        elif new_status == 'cancelled':
            # Send cancellation notices
            subject = f"❌ Appointment Cancelled - {appointment_date}"
            
            # To patient
            patient_html_message = render_to_string('appointments/emails/status_cancelled_patient.html', email_context)
            patient_plain_message = strip_tags(patient_html_message)
            
            send_mail(
                f"❌ Your Appointment Has Been Cancelled",
                patient_plain_message,
                settings.DEFAULT_FROM_EMAIL,
                [appointment.patient.email],
                html_message=patient_html_message,
                fail_silently=False,
            )
            
            # To doctor
            doctor_html_message = render_to_string('appointments/emails/status_cancelled_doctor.html', email_context)
            doctor_plain_message = strip_tags(doctor_html_message)
            
            send_mail(
                f"📅 Appointment Cancelled - {appointment.patient.get_full_name()}",
                doctor_plain_message,
                settings.DEFAULT_FROM_EMAIL,
                [appointment.doctor.email],
                html_message=doctor_html_message,
                fail_silently=False,
            )
            
        elif new_status == 'completed':
            # Send completion notice and review request
            patient_subject = f"⭐ Appointment Completed - How Was Your Experience?"
            patient_html_message = render_to_string('appointments/emails/status_completed.html', email_context)
            patient_plain_message = strip_tags(patient_html_message)
            
            send_mail(
                patient_subject,
                patient_plain_message,
                settings.DEFAULT_FROM_EMAIL,
                [appointment.patient.email],
                html_message=patient_html_message,
                fail_silently=False,
            )
        
        logger.info(f"Appointment status update emails sent for appointment ID: {appointment.id} - {old_status} → {new_status}")
        
    except Exception as e:
        logger.error(f"Failed to send appointment status update emails: {str(e)}")