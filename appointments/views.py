from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from .models import Appointment, TimeSlot
from .forms import AppointmentForm, TimeSlotForm
from doctors.models import Doctor
from ai_predictions.ml_models.no_show_model import no_show_predictor

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
        new_status = request.POST.get('status')
        if new_status in dict(Appointment.STATUS_CHOICES):
            appointment.status = new_status
            appointment.save()
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
                
                messages.success(request, 'Appointment booked successfully!')
                return redirect('appointments:appointments_list')
    else:
        form = AppointmentForm()
    
    context = {
        'form': form,
        'doctor': doctor,
    }
    return render(request, 'appointments/book_appointment.html', context)