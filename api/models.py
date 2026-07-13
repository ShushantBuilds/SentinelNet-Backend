from django.db import models
from django.contrib.auth.models import User

class OperatorProfile(models.Model):
    # Links this profile directly to the authenticated JWT user
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    # Financial thresholds for the AI Advisory Engine to check against
    monthly_budget_limit = models.DecimalField(max_digits=12, decimal_places=2, default=5000.00)
    high_risk_threshold = models.FloatField(default=60.0) # Flag anything above 60% risk

    def __str__(self):
        return f"{self.user.username}'s Profile"

class TransactionRecord(models.Model):
    # Foreign key linking every transaction to the specific user who ran it
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='transactions')
    
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    risk_score = models.FloatField()
    is_fraud = models.BooleanField()
    
    # Automatically stamps the exact date and time the database receives the record
    timestamp = models.DateTimeField(auto_now_add=True) 

    def __str__(self):
        status = "BLOCKED" if self.is_fraud else "APPROVED"
        return f"${self.amount} - {status} ({self.user.username})"