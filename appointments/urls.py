from django.urls import path
from . import views

app_name = 'appointments'

urlpatterns = [
    path('', views.appointments_list, name='appointments_list'),
    path('book/<int:doctor_id>/', views.book_appointment, name='book_appointment'),
    path('<int:appointment_id>/', views.appointment_detail, name='appointment_detail'),
    path('<int:appointment_id>/update-status/', views.update_appointment_status, name='update_appointment_status'),
]