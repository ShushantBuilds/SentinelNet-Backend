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
            shield_status = "safe"
            
            if is_fraud_detected or probability > 0.70:
                shield_status = "danger"
                advisory_message = "CRITICAL WARNING: High risk of phishing or fake storefront. Do not enter your credit card details on this page."
            elif probability > 0.30:
                shield_status = "warning"
                advisory_message = "CAUTION: This store was flagged for suspicious activity or a newly registered domain. Verify the merchant before paying."
            else:
                shield_status = "safe"
                advisory_message = "SECURE: This checkout page matches verified merchant patterns. It is safe to proceed."
            
            return Response({
                "fraud_prediction": int(prediction),
                "fraud_probability": round(float(probability), 4),
                "shield_status": shield_status, # Send status to React
                "advisory": advisory_message,
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
                    You are the SentinelNet Checkout Shield, an AI built to protect everyday consumers from online shopping scams.
                    The user is about to make a payment. You analyze e-commerce websites, merchant data, and checkout patterns for scam indicators (e.g., brand new domains, fake countdown timers, unencrypted gateways).
                    Explain the exact risks to the shopper in plain, urgent language. 
                    Refuse any queries unrelated to online shopping safety, e-commerce, or payment fraud.
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
                page_context = request.data.get('page_context', 'No webpage context extracted.')
                url = request.data.get('url', 'Unknown site')
                
                # Bundle the active webpage context into the prompt
                prompt = f"""
                The user is currently viewing the website: {url}
                
                --- WEBPAGE CONTENT START ---
                {page_context}
                --- WEBPAGE CONTENT END ---
                
                User Query: "{message}"
                """
                
                chat_rules = """
                You are SentinelNet Safety Assistant. You have real-time visibility into the exact webpage the user is looking at.
                Use the provided webpage content to answer questions, analyze discount claims, check for hidden terms, or detect scam red flags.
                If the user asks about the store, analyze the page text and provide concise, protective advice.
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

            # --- NEW: DEAL ANALYTICS ENGINE WITH LIVE SEARCH ---
            elif action_type == 'analyze_deal':
                product_url = request.data.get('product_url', '')
                page_context = request.data.get('page_context', '')
                
                prompt = f"""
                The user is considering buying a product at this link: {product_url}
                Here is the scraped text from their current checkout page:
                ---
                {page_context}
                ---
                
                Using your Google Search tool, look up the product to find its average retail price, and search the domain name of the store for recent reviews.
                
                Provide a brief, bulleted verdict covering:
                1. Store Legitimacy: Are there recent scam reports or bad reviews about this specific domain?
                2. Price History: What is the normal price of this product on major sites (like Amazon)? Is this current deal actually good, or a fake markup discount?
                3. Page Red Flags: Based on the scraped text, are there hidden fees, strict return policies, or fake FOMO tactics (e.g., "Only 1 left!")?
                4. Final Opinion: Is it safe and financially smart to complete this transaction right now?
                """
                
                deal_rules = """
                You are the SentinelNet Deal Analyst. Your job is to give users highly critical, objective advice on whether they should complete an online purchase. 
                You MUST use Google Search to verify the store's reputation and the product's actual market value before answering.
                Be concise, direct, and highly skeptical of "too good to be true" discounts.
                """
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=deal_rules,
                        temperature=0.2,
                        tools=[{"google_search": {}}] 
                    )
                )
                
                final_response_text = response.text
                
                # Append a verification badge if the search tool was successfully utilized
                if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
                    if hasattr(response.candidates[0].grounding_metadata, 'grounding_chunks') and response.candidates[0].grounding_metadata.grounding_chunks:
                        final_response_text += "\n\nSources verified via Google Search."

                return Response({"response": final_response_text}, status=status.HTTP_200_OK)
                
            return Response({"error": "Invalid action type specified."}, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            return Response({"error": f"Google API Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)