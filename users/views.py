from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegistrationForm, UserUpdateForm, ProfileUpdateForm
from appointments.models import Appointment
from doctors.models import Doctor
from django.utils import timezone

def home(request):
    return render(request, 'users/home.html')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            
            # Authenticate and login the user
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'Account created successfully! Welcome, {user.get_full_name()}!')
                return redirect('dashboard')
            else:
                messages.error(request, 'There was an error logging in after registration.')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'users/register.html', {'form': form})

def custom_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.get_full_name()}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'users/login.html')

def dashboard(request):
    user = request.user
    context = {}
    
    if user.profile.role == 'patient':
        upcoming_appointments = Appointment.objects.filter(patient=user, status__in=['pending', 'confirmed']).order_by('-date', '-time')[:3]
        appointments = Appointment.objects.filter(patient=user).order_by('-date', '-time')[:5]
        
        context.update({
            'appointments': appointments,
            'upcoming_appointments': upcoming_appointments,
        })
    
    elif user.profile.role == 'doctor':
        try:
            doctor_profile = Doctor.objects.get(user=user)
            today_appointments = Appointment.objects.filter(doctor=user, date__gte=timezone.now().date()).order_by('-time')[:5]
            appointments = Appointment.objects.filter(doctor=user).order_by('-date', '-time')[:5]
            
            context.update({
                'doctor_profile': doctor_profile,
                'appointments': appointments,
                'today_appointments': today_appointments,
            })
        except Doctor.DoesNotExist:
            pass
    
    return render(request, 'users/dashboard.html', context)

def logout_user(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('/')
@login_required
def profile(request):
    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=request.user)
        profile_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        user_form = UserUpdateForm(instance=request.user)
        profile_form = ProfileUpdateForm(instance=request.user.profile)
    
    context = {
        'user_form': user_form,
        'profile_form': profile_form,
    }
    
    return render(request, 'users/profile.html', context)