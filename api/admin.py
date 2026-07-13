from django.contrib import admin
from .models import *
# Register your models here.


admin.site.register(OperatorProfile)
@admin.register(TransactionRecord)
class TransactionRecordAdmin(admin.ModelAdmin):
    # This makes the admin table much easier to read
    list_display = ('amount', 'is_fraud', 'risk_score', 'user', 'timestamp')
    list_filter = ('is_fraud', 'user')