# forms.py
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Profile
from doctors.models import Doctor

class UserRegistrationForm(UserCreationForm):
    # Profile fields
    role = forms.ChoiceField(choices=Profile.ROLE_CHOICES, widget=forms.Select(attrs={'class': 'form-control', 'id': 'role-select'}))
    phone = forms.CharField(required=False, widget=forms.TextInput(attrs={'class': 'form-control'}))
    date_of_birth = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}))
    address = forms.CharField(required=False, widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
    profile_picture = forms.ImageField(required=False, widget=forms.FileInput(attrs={'class': 'form-control'}))
    
    # Doctor fields (optional initially)
    specialization = forms.ChoiceField(choices=Doctor.SPECIALIZATION_CHOICES, required=False, widget=forms.Select(attrs={'class': 'form-control doctor-field'}))
    experience = forms.IntegerField(required=False, min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control doctor-field'}))
    about = forms.CharField(required=False, widget=forms.Textarea(attrs={'class': 'form-control doctor-field', 'rows': 3}))
    consultation_fee = forms.DecimalField(required=False, min_value=0, decimal_places=2, widget=forms.NumberInput(attrs={'class': 'form-control doctor-field'}))
    license_number = forms.CharField(required=False, widget=forms.TextInput(attrs={'class': 'form-control doctor-field'}))
    
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        role = cleaned_data.get('role')
        
        # Validate doctor fields if role is doctor
        if role == 'doctor':
            specialization = cleaned_data.get('specialization')
            experience = cleaned_data.get('experience')
            consultation_fee = cleaned_data.get('consultation_fee')
            license_number = cleaned_data.get('license_number')
            
            if not specialization:
                self.add_error('specialization', 'Specialization is required for doctors')
            if not experience:
                self.add_error('experience', 'Experience is required for doctors')
            if not consultation_fee:
                self.add_error('consultation_fee', 'Consultation fee is required for doctors')
            if not license_number:
                self.add_error('license_number', 'License number is required for doctors')
        
        return cleaned_data
    
    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()
            
            # Save profile
            profile = Profile.objects.get(user=user)
            profile.role = self.cleaned_data.get('role')
            profile.phone = self.cleaned_data.get('phone')
            profile.date_of_birth = self.cleaned_data.get('date_of_birth')
            profile.address = self.cleaned_data.get('address')
            
            if self.cleaned_data.get('profile_picture'):
                profile.profile_picture = self.cleaned_data.get('profile_picture')
            
            profile.save()
            
            # Create doctor profile if role is doctor
            if self.cleaned_data.get('role') == 'doctor':
                Doctor.objects.create(
                    user=user,
                    specialization=self.cleaned_data.get('specialization'),
                    experience=self.cleaned_data.get('experience'),
                    about=self.cleaned_data.get('about', ''),
                    consultation_fee=self.cleaned_data.get('consultation_fee'),
                    license_number=self.cleaned_data.get('license_number'),
                    is_available=True
                )
        
        return user


class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={
        'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
    }))
    first_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={
        'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
    }))
    last_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={
        'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
    }))

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
        }

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['phone', 'date_of_birth', 'address', 'profile_picture']
        widgets = {
            'phone': forms.TextInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'date_of_birth': forms.DateInput(attrs={
                'type': 'date',
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'address': forms.Textarea(attrs={
                'rows': 3,
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
            'profile_picture': forms.FileInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500'
            }),
        }