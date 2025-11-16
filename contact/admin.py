from django.contrib import admin
from .models import ContactCategory, ContactMessage, FAQ, SupportTicket
# Register your models here.
admin.site.register(ContactMessage)
admin.site.register(ContactCategory)
admin.site.register(FAQ)
admin.site.register(SupportTicket)