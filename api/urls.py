from django.urls import path
from .views import *

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health_check'),
    path('analyze/', AnalyzeCheckoutView.as_view(), name='analyze_checkout'),
    path('register/', RegisterView.as_view(), name='register'),
    path('profile/', ManageProfileView.as_view(), name='profile'),
    path('ai/', AiAssistantView.as_view(), name='ai_assistant'),
]