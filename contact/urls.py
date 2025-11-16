from django.urls import path
from . import views

app_name = 'contact'

urlpatterns = [
    # Public pages
    path('', views.contact_page, name='contact_page'),
    path('success/', views.contact_success, name='contact_success'),
    path('faqs/', views.faq_list, name='faq_list'),
    path('my-messages/', views.my_contact_messages, name='my_contact_messages'),
    
    # Admin pages
    path('admin/', views.contact_admin, name='contact_admin'),
    path('admin/message/<int:message_id>/', views.contact_message_detail, name='contact_message_detail'),
    path('admin/message/<int:message_id>/update-status/', views.update_message_status, name='update_message_status'),
    
    # API endpoints
    path('api/faqs/', views.get_faq_by_category, name='get_faq_by_category'),
]