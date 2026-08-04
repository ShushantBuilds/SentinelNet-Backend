import os
import json
import numpy as np

# GOOGLE GEMINI IMPORTS
from google import genai
from google.genai import types

# MISTRAL & TAVILY IMPORTS
from mistralai.client import Mistral
from tavily import TavilyClient

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.contrib.auth.models import User
from django.db.models import Sum

from .apps import ApiConfig
from .serializers import UserSerializer
from .models import *

# --- INITIALIZE AI CLIENTS ---
client = genai.Client()  # Uses GEMINI_API_KEY from environment
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


def search_web_for_deal(query: str) -> str:
    """Helper function to execute the Tavily search for Mistral AI."""
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        formatted_results = [
            f"Source: {res['url']}\nContent: {res['content']}" 
            for res in response.get('results', [])
        ]
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Search failed: {str(e)}"


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
                "shield_status": shield_status,
                "advisory": advisory_message,
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
        
        if hasattr(profile, 'total_accumulated_spend'):
            profile.total_accumulated_spend = 0
            profile.save()
            
        return Response({
            "message": "Total accumulated spend has been reset.",
            "total_accumulated_spend": 0
        }, status=status.HTTP_200_OK)


class AiAssistantView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        action_type = request.data.get('type')
        
        try:
            # ==========================================
            # ENGINE 1: GEMINI (Used for Insight & Chat)
            # ==========================================
            if action_type == 'insight':
                query = request.data.get('query')
                history = request.data.get('history', [])
                budget = request.data.get('budget', 0)
                
                prompt = f"""
                The operator has asked: "{query}"
                Current System Context: Target Budget: ${budget}, Recent Transactions: {history}
                Provide a brief, analytical response. No markdown headers.
                """

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
                        temperature=0.2
                    )
                )
                return Response({"response": response.text}, status=status.HTTP_200_OK)
                
            elif action_type == 'chat':
                message = request.data.get('message')
                page_context = request.data.get('page_context', 'No webpage context extracted.')
                url = request.data.get('url', 'Unknown site')
                
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

            # ==========================================
            # ENGINE 2: MISTRAL + TAVILY (Deal Analysis)
            # ==========================================
            elif action_type == 'analyze_deal':
                product_url = request.data.get('product_url', '')
                page_context = request.data.get('page_context', '')
                
                prompt = f"""
                The user is considering buying a product at this link: {product_url}
                Here is the scraped text from their current checkout page:
                ---
                {page_context}
                ---
                """
                
                deal_rules = """
                You are the SentinelNet Deal Analyst. Your job is to give users highly critical, objective advice on whether they should complete an online purchase. 
                Use the 'search_web' tool to verify the store's reputation and the product's actual market value.
                Provide a brief, bulleted verdict covering: Store Legitimacy, Price History, Page Red Flags, and a Final Opinion.
                """
                
                tools = [{
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Search the live web for product price history, store reviews, and scam reports.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query (e.g., 'Is [store name] legit?' or '[product name] average price')."
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }]
                
                messages = [
                    {"role": "system", "content": deal_rules},
                    {"role": "user", "content": prompt}
                ]
                
                response = mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=messages,
                    tools=tools,
                    temperature=0.2
                )
                
                response_message = response.choices[0].message
                messages.append(response_message)
                
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "search_web":
                            args = json.loads(tool_call.function.arguments)
                            search_results = search_web_for_deal(args["query"])
                            
                            messages.append({
                                "role": "tool",
                                "name": "search_web",
                                "content": search_results,
                                "tool_call_id": tool_call.id
                            })
                            
                    final_response = mistral_client.chat.complete(
                        model="mistral-small-latest",
                        messages=messages,
                        temperature=0.2
                    )
                    final_text = final_response.choices[0].message.content + "\n\nSources verified via Tavily Search."
                else:
                    final_text = response_message.content
                    
                return Response({"response": final_text}, status=status.HTTP_200_OK)
                
            return Response({"error": "Invalid action type specified."}, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            return Response({"error": f"AI Assistant Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)