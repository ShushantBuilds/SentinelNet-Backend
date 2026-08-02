from google import genai
from google.genai import types
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

client = genai.Client()

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
            is_fraud_detected = bool(prediction > 0.30)

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
            traffic_light = "green"
            
            if is_fraud_detected or probability > 0.60:
                traffic_light = "red"
                advisory_message = "DO NOT SHIP. High probability of chargeback or fake payment receipt. Cancel order or demand wire transfer."
            elif probability > 0.25:
                traffic_light = "yellow"
                advisory_message = "CAUTION: Unusual pattern detected. Verify the buyer's identity or contact them directly before fulfilling this order."
            else:
                traffic_light = "green"
                advisory_message = "CLEARED: Transaction matches safe customer patterns. Proceed with fulfillment."
            
            return Response({
                "fraud_prediction": int(prediction),
                "fraud_probability": round(float(probability), 4),
                "traffic_light": traffic_light, # Send the color to React
                "advisory": advisory_message,
                "total_spent": float(total_spent) # We will rename this to "Total Verified Sales" in React
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,) 
    serializer_class = UserSerializer

class ManageProfileView(APIView):
    permission_classes = [IsAuthenticated] # Must be logged in

    def get(self, request):
        profile = request.user.profile
        return Response({
            "monthly_budget_limit": profile.monthly_budget_limit,
            "total_accumulated_spend": getattr(profile, 'total_accumulated_spend', 0)
        })

    def post(self, request):
        profile = request.user.profile
        new_budget = request.data.get('monthly_budget_limit')
        
        if new_budget:
            profile.monthly_budget_limit = new_budget
            profile.save()
            return Response({"message": "Budget updated successfully."}, status=status.HTTP_200_OK)
        return Response({"error": "Invalid data."}, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        """Resets the accumulated spend metrics for the user"""
        profile = request.user.profile

        TransactionRecord.objects.filter(user=request.user).delete()
        
        # 1. Reset the tracking field to 0 if stored directly on the profile
        if hasattr(profile, 'total_accumulated_spend'):
            profile.total_accumulated_spend = 0
            profile.save()
            
        # 2. OPTIONAL: If spend is calculated from a Transaction model, clear them out:
        # request.user.transactions.all().delete()
        
        return Response({
            "message": "Total accumulated spend has been reset.",
            "total_accumulated_spend": 0
        }, status=status.HTTP_200_OK)

class AiAssistantView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        action_type = request.data.get('type')
        
        try:
            if action_type == 'insight':
                query = request.data.get('query')
                history = request.data.get('history', [])
                budget = request.data.get('budget', 0)
                
                # The user's specific request
                prompt = f"""
                The operator has asked: "{query}"
                Current System Context: Target Budget: ${budget}, Recent Transactions: {history}
                Provide a brief, analytical response. No markdown headers.
                """

                # THE GUARDRAIL: Strict rules for the Insight Engine
                insight_rules = """
                    You are the AI core of SentinelNet SME, a protection portal for small business owners and freelancers.
                    You analyze incoming orders, revenue targets, and fraud risks (like chargebacks or fake receipts).
                    Explain risk factors in simple, non-technical terms. 
                    If an order is high risk, suggest practical steps (e.g., "Ask for ID verification", "Wait for funds to clear before shipping").
                    Refuse any queries unrelated to business operations or fraud protection.
                    """
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=insight_rules,
                        temperature=0.2 # Lower temperature makes the AI more strict and analytical
                    )
                )
                return Response({"response": response.text}, status=status.HTTP_200_OK)
                
            elif action_type == 'chat':
                message = request.data.get('message')
                prompt = message
                
                # THE GUARDRAIL: Strict rules for the Chatbot
                chat_rules = """
                    You are SentinelNet Merchant Support. You help small business owners verify orders and protect against payment fraud.
                    Keep your answers brief, empathetic, and actionable. Avoid deep technical jargon like "feature vectors."
                    """
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=chat_rules,
                        temperature=0.3
                    )
                )
                return Response({"response": response.text}, status=status.HTTP_200_OK)
                
            return Response({"error": "Invalid action type specified."}, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            return Response({"error": f"Google API Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)