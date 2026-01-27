from django.contrib import admin

# Register your models here.
from .models import SymptomCheckHistory, SymptomPattern, DiseasePattern, AIModelPerformance
admin.site.register(SymptomCheckHistory)
admin.site.register(SymptomPattern)
admin.site.register(DiseasePattern)
admin.site.register(AIModelPerformance)