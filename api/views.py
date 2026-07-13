from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from .apps import ApiConfig
import numpy as np
from .serializers import UserSerializer
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.contrib.auth.models import User

# NEW IMPORTS FOR THE AI ADVISORY ENGINE
from django.db.models import Sum
from .models import *

class HealthCheckView(APIView):
    def get(self, request):
        return Response({
            "status": "online", 
            "model_loaded": ApiConfig.rf_model is not None
        }, status=status.HTTP_200_OK)

class PredictFraudView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            features = request.data.get('features')
            
            if not features or len(features) != 30:
                return Response(
                    {"error": "Invalid payload. Must provide exactly 30 features."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            amount = float(features[0])
            feature_array = np.array(features).reshape(1, -1)
            
            prediction = ApiConfig.rf_model.predict(feature_array)[0]
            probability = ApiConfig.rf_model.predict_proba(feature_array)[0][1]
            is_fraud_detected = bool(prediction == 1)

            TransactionRecord.objects.create(
                user=request.user,
                amount=amount,
                risk_score=probability * 100,
                is_fraud=is_fraud_detected
            )

            # THE FIX: Safely get the profile, or create it if it's missing!
            profile, created = OperatorProfile.objects.get_or_create(
                user=request.user, 
                defaults={'monthly_budget_limit': 5000.00}
            )
            
            budget = float(profile.monthly_budget_limit)

            total_spent_raw = TransactionRecord.objects.filter(
                user=request.user,
                is_fraud=False
            ).aggregate(Sum('amount'))['amount__sum'] or 0.0

            total_spent = float(total_spent_raw)

            advisory_message = ""
            
            if is_fraud_detected:
                advisory_message = "Blocked: Anomalous data patterns detected in metadata matrix."
            else:
                if total_spent > budget:
                    overage = total_spent - budget
                    advisory_message = f"Warning: This purchase pushes you ${overage:,.2f} over your monthly budget of ${budget:,.2f}."
                else:
                    remaining = budget - total_spent
                    advisory_message = f"Secure: You have ${remaining:,.2f} remaining in your monthly budget."
            
            return Response({
                "fraud_prediction": int(prediction),
                "fraud_probability": round(float(probability), 4),
                "advisory": advisory_message,
                "total_spent": float(total_spent)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,) 
    serializer_class = UserSerializer

class ManageProfileView(APIView):
    permission_classes = [IsAuthenticated] 

    def get(self, request):
        # THE FIX: Safely get or create the profile
        profile, created = OperatorProfile.objects.get_or_create(
            user=request.user, 
            defaults={'monthly_budget_limit': 5000.00}
        )
        return Response({"monthly_budget_limit": profile.monthly_budget_limit})

    def post(self, request):
        # THE FIX: Safely get or create the profile
        profile, created = OperatorProfile.objects.get_or_create(
            user=request.user, 
            defaults={'monthly_budget_limit': 5000.00}
        )
        new_budget = request.data.get('monthly_budget_limit')
        
        if new_budget:
            profile.monthly_budget_limit = new_budget
            profile.save()
            return Response({"message": "Budget updated successfully"})
            
        return Response({"error": "No budget provided"}, status=400)