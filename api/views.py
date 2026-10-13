import os
import re
import json
import numpy as np

from google import genai
from google.genai import types

from groq import Groq
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
groq_client = Groq(api_key=os.environ.get("PROJECT_API"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# Model names live in env vars so a retired model only needs a config change.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")  # verify against Google's current model list

MAX_SEARCH_ROUNDS = 2


def search_web_for_deal(query: str) -> str:
    """Helper function to execute the Tavily search for the LLM."""
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        formatted_results = [
            f"Source: {res['url']}\nContent: {res['content']}"
            for res in response.get('results', [])
        ]
        return "\n\n".join(formatted_results) or "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"


def run_llm_analysis(messages, tools):
    """Let the model search up to N times, then force a JSON answer with no tools."""
    for _ in range(MAX_SEARCH_ROUNDS):
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""  # model answered directly

        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                result = search_web_for_deal(args["query"])
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            })

    # Final pass: no tools allowed, JSON only
    messages.append({
        "role": "user",
        "content": (
            "Searching is finished. Do NOT call any tools. Respond now with only the JSON object "
            "containing \"analysis\" and \"llm_risk_score\"."
        ),
    })
    final = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return final.choices[0].message.content or ""


def parse_llm_json(text: str):
    """Pull the first JSON object out of the text, tolerating code fences."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON found", text, 0)
    return json.loads(match.group(0))


class HealthCheckView(APIView):
    def get(self, request):
        return Response({
            "status": "online",
            "model_loaded": ApiConfig.rf_model is not None
        }, status=status.HTTP_200_OK)


class AnalyzeCheckoutView(APIView):
    """
    Blends the Random Forest ML prediction (30%) with Groq+Tavily Live Web Analysis (70%).
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            url = request.data.get('url', 'Unknown URL')
            page_context = request.data.get('page_context', '')
            amount = float(request.data.get('amount', 0.0))

            # ==========================================
            # ENGINE 1: RANDOM FOREST ML (30% Weight)
            # ==========================================
            dummy_pca = [0.0] * 29
            feature_array = np.array([amount] + dummy_pca).reshape(1, -1)

            ml_probability = ApiConfig.rf_model.predict_proba(feature_array)[0][1]
            ml_risk_score = int(ml_probability * 100)

            # ==========================================
            # ENGINE 2: GROQ + TAVILY (70% Weight)
            # ==========================================
            system_rules = """
            You are a cybersecurity agent. Analyze the store reputation using the web search tool.
            You MUST return your final response strictly as a JSON object with two keys:
            1. "analysis": A 2-3 paragraph markdown string explaining the seller's reputation and red flags.
            2. "llm_risk_score": An integer from 0 (completely safe) to 100 (confirmed scam).
            """

            prompt = f"Product URL: {url}\nScraped Page Context: {page_context}\nTransaction Amount: ${amount}"

            tools = [{
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the live web for product price history, store reviews, and scam reports.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query (e.g., 'Is [store name] legit?')."}
                        },
                        "required": ["query"],
                    },
                },
            }]

            messages = [
                {"role": "system", "content": system_rules},
                {"role": "user", "content": prompt}
            ]

            final_content = run_llm_analysis(messages, tools)

            try:
                llm_data = parse_llm_json(final_content)
                llm_risk_score = max(0, min(100, int(llm_data.get('llm_risk_score', 50))))
                seller_analysis = llm_data.get('analysis', 'Analysis failed.')
            except (json.JSONDecodeError, ValueError, TypeError):
                llm_risk_score = 50
                seller_analysis = f"AI formatting error. Raw output: {final_content}"

            # ==========================================
            # ENGINE 3: THE BLENDED VERDICT
            # ==========================================
            final_blended_score = int((llm_risk_score * 0.7) + (ml_risk_score * 0.3))

            if final_blended_score > 70:
                shield_status = "danger"
                is_fraud_detected = True
            elif final_blended_score > 40:
                shield_status = "warning"
                is_fraud_detected = False
            else:
                shield_status = "safe"
                is_fraud_detected = False

            # Log the transaction with the new blended risk score
            TransactionRecord.objects.create(
                user=request.user,
                amount=amount,
                risk_score=final_blended_score,
                is_fraud=is_fraud_detected
            )

            # Ensure budget profile exists
            OperatorProfile.objects.get_or_create(
                user=request.user,
                defaults={'monthly_budget_limit': 5000.00}
            )

            return Response({
                "final_risk_score": final_blended_score,
                "shield_status": shield_status,
                "seller_analysis": seller_analysis,
                "ml_contribution": ml_risk_score,
                "llm_contribution": llm_risk_score
            }, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            traceback.print_exc()  # full traceback in the Django terminal
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
            # GEMINI (Used for Insight & Chat)
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
                    The user is about to make a payment. You analyze e-commerce websites, merchant data, and checkout patterns for scam indicators.
                    Explain the exact risks to the shopper in plain, urgent language.
                    Refuse any queries unrelated to online shopping safety, e-commerce, or payment fraud.
                """

                response = client.models.generate_content(
                    model=GEMINI_MODEL,
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
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=chat_rules,
                        temperature=0.3
                    )
                )
                return Response({"response": response.text}, status=status.HTTP_200_OK)

            return Response({"error": "Invalid action type specified."}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": f"AI Assistant Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)