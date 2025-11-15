from django.contrib import admin
from .models import Appointment, TimeSlot
# Register your models here.
admin.site.register(Appointment)
admin.site.register(TimeSlot)