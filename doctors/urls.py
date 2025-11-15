from django.urls import path
from . import views

app_name = 'doctors'

urlpatterns = [
    path('', views.doctors_list, name='doctors_list'),
    path('<int:doctor_id>/', views.doctor_detail, name='doctor_detail'),
    path('dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('profile/update/', views.update_doctor_profile, name='update_doctor_profile'),
    path('availability/', views.manage_availability, name='manage_availability'),
    path('appointments/', views.doctor_appointments, name='doctor_appointments'),
    path('reviews/', views.doctor_reviews, name='doctor_reviews'),
    # Add this view for toggling availability
    path('toggle-availability/', views.toggle_availability, name='toggle_availability'),
]