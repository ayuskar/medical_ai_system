from django.shortcuts import render, redirect

# Create your views here.
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.db.models import Count, Avg, Q
from .models import Doctor, Availability
from .forms import DoctorForm, AvailabilityForm
from appointments.models import Appointment
from reviews.models import Review
from datetime import datetime, timedelta
@login_required
def toggle_availability(request):
    if not hasattr(request.user, 'doctor'):
        messages.error(request, 'You are not registered as a doctor.')
        return redirect('users:dashboard')
    
    doctor = request.user.doctor
    doctor.is_available = not doctor.is_available
    doctor.save()
    
    status = "available" if doctor.is_available else "not available"
    messages.success(request, f'You are now {status} for appointments.')
    
    return redirect('doctors:doctor_dashboard')
def doctors_list(request):
    doctors = Doctor.objects.filter(is_available=True).select_related('user')
    
    # Filter by specialization
    specialization = request.GET.get('specialization')
    if specialization:
        doctors = doctors.filter(specialization=specialization)
    
    # Search by name or specialization
    search_query = request.GET.get('search')
    if search_query:
        doctors = doctors.filter(
            Q(user__first_name__icontains=search_query) |
            Q(user__last_name__icontains=search_query) |
            Q(specialization__icontains=search_query)
        )
    
    context = {
        'doctors': doctors,
        'specializations': Doctor.SPECIALIZATION_CHOICES,
    }
    return render(request, 'doctors/doctors_list.html', context)

def doctor_detail(request, doctor_id):
    doctor = get_object_or_404(Doctor, id=doctor_id)
    reviews = Review.objects.filter(appointment__doctor=doctor.user).select_related('appointment__patient')
    availabilities = doctor.availabilities.filter(is_available=True)
    
    context = {
        'doctor': doctor,
        'reviews': reviews,
        'availabilities': availabilities,
    }
    return render(request, 'doctors/doctor_detail.html', context)

@login_required
def doctor_dashboard(request):
    if not hasattr(request.user, 'doctor'):
        messages.error(request, 'You are not registered as a doctor.')
        return redirect('dashboard')
    
    doctor = request.user.doctor
    appointments = doctor.user.doctor_appointments.all().order_by('-date', '-time')[:10]
    today_appointments = appointments.filter(date__date=timezone.now().date())
    
    context = {
        'doctor': doctor,
        'appointments': appointments,
        'today_appointments': today_appointments,
    }
    return render(request, 'doctors/doctor_dashboard.html', context)

@login_required
def doctor_dashboard(request):
    if not hasattr(request.user, 'doctor'):
        messages.error(request, 'You are not registered as a doctor.')
        return redirect('users:dashboard')
    
    doctor = request.user.doctor
    
    # Get today's date
    today = timezone.now().date()
    
    # Appointments data
    appointments = Appointment.objects.filter(doctor=request.user)
    today_appointments = appointments.filter(date=today).order_by('time')
    upcoming_appointments = appointments.filter(date__gte=today, status__in=['pending', 'confirmed']).order_by('date', 'time')[:5]
    
    # Statistics
    total_appointments = appointments.count()
    completed_appointments = appointments.filter(status='completed').count()
    pending_appointments = appointments.filter(status='pending').count()
    
    # Weekly appointments for chart
    start_date = today - timedelta(days=30)
    weekly_appointments = appointments.filter(
        date__gte=start_date
    ).values('date').annotate(count=Count('id')).order_by('date')
    
    # Reviews and ratings
    reviews = Review.objects.filter(appointment__doctor=request.user)
    average_rating = reviews.aggregate(Avg('rating'))['rating__avg'] or 0
    total_reviews = reviews.count()
    
    # High-risk appointments (no-show prediction > 70%)
    high_risk_appointments = appointments.filter(
        prediction_score__gt=0.7,
        status__in=['pending', 'confirmed'],
        date__gte=today
    ).order_by('date', 'time')
    
    context = {
        'doctor': doctor,
        'today_appointments': today_appointments,
        'upcoming_appointments': upcoming_appointments,
        'total_appointments': total_appointments,
        'completed_appointments': completed_appointments,
        'pending_appointments': pending_appointments,
        'average_rating': round(average_rating, 1),
        'total_reviews': total_reviews,
        'high_risk_appointments': high_risk_appointments,
        'weekly_appointments': list(weekly_appointments),
    }
    return render(request, 'doctors/doctor_dashboard.html', context)

@login_required
def update_doctor_profile(request):
    if not hasattr(request.user, 'doctor'):
        messages.error(request, 'You are not registered as a doctor.')
        return redirect('users:dashboard')
    
    doctor = request.user.doctor
    
    if request.method == 'POST':
        form = DoctorForm(request.POST, instance=doctor)
        if form.is_valid():
            form.save()
            messages.success(request, 'Doctor profile updated successfully!')
            return redirect('doctors:doctor_dashboard')
    else:
        form = DoctorForm(instance=doctor)
    
    context = {
        'form': form,
        'doctor': doctor,
    }
    return render(request, 'doctors/update_doctor_profile.html', context)

@login_required
def manage_availability(request):
    if not hasattr(request.user, 'doctor'):
        messages.error(request, 'You are not registered as a doctor.')
        return redirect('users:dashboard')
    
    doctor = request.user.doctor
    availabilities = doctor.availabilities.all()
    
    if request.method == 'POST':
        form = AvailabilityForm(request.POST)
        if form.is_valid():
            availability = form.save(commit=False)
            availability.doctor = doctor
            
            # Check if availability for this day already exists
            existing_availability = Availability.objects.filter(
                doctor=doctor, 
                day=availability.day
            ).first()
            
            if existing_availability:
                existing_availability.start_time = availability.start_time
                existing_availability.end_time = availability.end_time
                existing_availability.is_available = availability.is_available
                existing_availability.save()
                messages.success(request, f'Availability for {availability.get_day_display()} updated successfully!')
            else:
                availability.save()
                messages.success(request, f'Availability for {availability.get_day_display()} added successfully!')
            
            return redirect('doctors:manage_availability')
    else:
        form = AvailabilityForm()
    
    context = {
        'form': form,
        'availabilities': availabilities,
        'doctor': doctor,
    }
    return render(request, 'doctors/manage_availability.html', context)

@login_required
def doctor_appointments(request):
    if not hasattr(request.user, 'doctor'):
        messages.error(request, 'You are not registered as a doctor.')
        return redirect('users:dashboard')
    
    doctor = request.user.doctor
    appointments = Appointment.objects.filter(doctor=request.user).order_by('-date', '-time')
    
    # Filter by status
    status_filter = request.GET.get('status')
    if status_filter:
        appointments = appointments.filter(status=status_filter)
    
    context = {
        'appointments': appointments,
        'doctor': doctor,
    }
    return render(request, 'doctors/doctor_appointments.html', context)

@login_required
def doctor_reviews(request):
    if not hasattr(request.user, 'doctor'):
        messages.error(request, 'You are not registered as a doctor.')
        return redirect('users:dashboard')
    
    doctor = request.user.doctor
    reviews = Review.objects.filter(appointment__doctor=request.user).select_related('appointment__patient').order_by('-created_at')
    
    # Calculate rating distribution
    rating_distribution = {}
    for i in range(1, 6):
        rating_distribution[i] = reviews.filter(rating=i).count()
    
    # Sentiment analysis
    positive_reviews = reviews.filter(sentiment='POSITIVE').count()
    negative_reviews = reviews.filter(sentiment='NEGATIVE').count()
    neutral_reviews = reviews.filter(sentiment='NEUTRAL').count()
    
    context = {
        'reviews': reviews,
        'doctor': doctor,
        'rating_distribution': rating_distribution,
        'positive_reviews': positive_reviews,
        'negative_reviews': negative_reviews,
        'neutral_reviews': neutral_reviews,
    }
    return render(request, 'doctors/doctor_reviews.html', context)
