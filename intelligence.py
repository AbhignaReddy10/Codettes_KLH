import os
import requests


def generate_ai_plan(restock_alerts, surplus_items):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "ai_plan": "AI key not configured.",
            "confidence_score": 0
        }

    prompt = f"""
    You are an AI food supply chain analyst.

    Restock alerts: {restock_alerts}
    Surplus items: {surplus_items}

    Provide:
    1. Urgent restocking recommendations
    2. Redistribution strategy for surplus items
    3. Food waste reduction plan
    4. Strategic business insight
    """

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=body)

        if response.status_code != 200:
            return {
                "ai_plan": f"Gemini API error: {response.text}",
                "confidence_score": 0
            }

        result = response.json()

        if "candidates" not in result:
            return {
                "ai_plan": f"Unexpected Gemini response: {result}",
                "confidence_score": 0
            }

        text = result["candidates"][0]["content"]["parts"][0]["text"]

        confidence = min(
            100,
            60 + len(restock_alerts) * 10 + len(surplus_items) * 10
        )

        return {
            "ai_plan": text,
            "confidence_score": confidence
        }

    except Exception as e:
        return {
            "ai_plan": f"AI analysis failed: {str(e)}",
            "confidence_score": 0
        }
