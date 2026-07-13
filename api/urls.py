from django.urls import path
from .views import *

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health_check'),
    path('predict/', PredictFraudView.as_view(), name='predict_fraud'),
    path('register/', RegisterView.as_view(), name='register'),
    path('profile/', ManageProfileView.as_view(), name='profile'),
]