from django.contrib import admin

# Register your models here.
from .models import SymptomCheckHistory, SymptomPattern, AIPerformanceLog, DiseaseDatabase
admin.site.register(SymptomCheckHistory)
admin.site.register(SymptomPattern)
admin.site.register(AIPerformanceLog)
admin.site.register(DiseaseDatabase)