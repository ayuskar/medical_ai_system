from django.contrib import admin

# Register your models here.
from .models import Availability, Doctor
admin.site.register(Availability)
admin.site.register(Doctor)