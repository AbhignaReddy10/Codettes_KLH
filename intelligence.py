import os
import json
from google import genai
from typing import Dict
import pandas as pd
from datetime import datetime

# ============================================
# SECURITY: Use Environment Variables
# ============================================
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # Fallback for development (REMOVE IN PRODUCTION)
    API_KEY = "AIzaSyD0Tuqq3lJQmBHYG3E4KuH52ftb7une1Y4"
    print("⚠️  WARNING: Using hardcoded API key. Set GEMINI_API_KEY env var for production!")

client = genai.Client(api_key=API_KEY)


# ============================================
# GEMINI 3 FLASH INTEGRATION
# ============================================

def get_prescriptive_json(item_name: str, surplus_kg: float) -> Dict:
    """
    🤖 INTELLIGENCE ENGINE: Analyzes surplus food with Gemini 3 Flash

    Returns structured JSON with:
    - AI Reasoning (why surplus happened)
    - NGO Recommendation (nearest, best-fit)
    - Impact Metrics (CO2, meals, cost)
    - Handling Instructions (safety, shelf life)

    Parameters:
    -----------
    item_name : str
        Food item name (e.g., "Chicken Biryani", "Paneer")
    surplus_kg : float
        Surplus quantity in kilograms

    Returns:
    --------
    Dict with keys: reasoning, ngo_recommendation, impact_metrics, handling_instructions
    """

    MODEL_ID = "gemini-3-flash-preview"

    # ============================================
    # PROMPT ENGINEERING FOR CONSISTENCY
    # ============================================
    prompt = f"""You are a food waste expert AI for a smart restaurant kitchen management system.

TASK: Analyze surplus food and provide structured action plan.

CONTEXT:
- Item: {item_name}
- Surplus Quantity: {surplus_kg} kg
- Current Date: {datetime.now().strftime('%Y-%m-%d')}
- Restaurant Type: Cloud Kitchen / Restaurant (Indian cuisine)

ANALYSIS REQUIRED:
Return ONLY valid JSON (no markdown, no extra text).

1. "reasoning" - 2-3 sentences explaining why surplus might have occurred
   (Consider: demand fluctuations, over-ordering, cancellations, spoilage risks)

2. "ngo_recommendation" - Which local NGO should receive this food?
   Format: "Organization Name (Distance: Xkm away, Serves Y+ people daily)"
   (Choose from real NGOs: Smile Foundation, Akshaya Patra, RDF, Provide, etc.)

3. "impact_metrics" - Calculate environmental & social impact
   {{
       "co2_saved_kg": {surplus_kg} × 2.5,
       "meals_provided": {surplus_kg} × 10,
       "cost_saved_inr": {surplus_kg} × 250 (estimate based on item type)
   }}

4. "handling_instructions" - Safety & logistics
   Format: "Keep at 4°C | Consume within 6 hours | Transport in insulated box"
   (Adjust based on food type: {item_name})

5. "confidence_score" - Your confidence in this recommendation (0-100)

RESPONSE FORMAT (EXAMPLE):
{{
  "reasoning": "High stock due to wedding cancellations; oversized initial order",
  "ngo_recommendation": "Akshaya Patra (2.3km away, serves 50000+ meals daily)",
  "impact_metrics": {{
    "co2_saved_kg": 25.0,
    "meals_provided": 100,
    "cost_saved_inr": 2500
  }},
  "handling_instructions": "Keep at 4°C | Consume within 6 hours | Transport in sealed container",
  "confidence_score": 92
}}

Now analyze {item_name} ({surplus_kg}kg) and return ONLY JSON:
"""

    try:
        # Call Gemini 3 Flash with JSON response mode
        response = client.models.generate_content(
            model=MODEL_ID,
            config={
                'response_mime_type': 'application/json',
                'temperature': 0.7,
                'top_p': 0.95
            },
            contents=prompt
        )

        # Parse JSON response
        result = json.loads(response.text)

        # Validate required fields
        required_fields = ['reasoning', 'ngo_recommendation', 'impact_metrics', 'handling_instructions']
        for field in required_fields:
            if field not in result:
                result[field] = f"[Missing: {field}]"

        # Ensure impact metrics are numeric
        if isinstance(result.get('impact_metrics'), dict):
            metrics = result['impact_metrics']
            metrics['co2_saved_kg'] = float(metrics.get('co2_saved_kg', surplus_kg * 2.5))
            metrics['meals_provided'] = int(metrics.get('meals_provided', surplus_kg * 10))
            metrics['cost_saved_inr'] = float(metrics.get('cost_saved_inr', surplus_kg * 250))

        return result

    except json.JSONDecodeError as e:
        # Fallback if Gemini returns non-JSON
        return {
            "reasoning": f"Failed to parse AI response: {str(e)}",
            "ngo_recommendation": "Manual review required",
            "impact_metrics": {
                "co2_saved_kg": round(surplus_kg * 2.5, 2),
                "meals_provided": int(surplus_kg * 10),
                "cost_saved_inr": round(surplus_kg * 250, 2)
            },
            "handling_instructions": "Contact restaurant manager for manual handling",
            "confidence_score": 0,
            "error": "JSON parsing failed"
        }

    except Exception as e:
        return {
            "error": str(e),
            "tip": "Check API key and Gemini 3 Flash model availability",
            "item": item_name,
            "status": "error"
        }


# ============================================
# OPTIONAL: Batch Analysis (For CSV uploads)
# ============================================

def analyze_surplus_batch(surplus_items: list) -> list:
    """
    Analyze multiple surplus items in batch.

    Parameters:
    -----------
    surplus_items : List[Dict]
        List of {"item_name": str, "surplus_kg": float}

    Returns:
    --------
    List of analysis results
    """
    results = []
    for item in surplus_items:
        result = get_prescriptive_json(item['item_name'], item['surplus_kg'])
        results.append({
            "item": item['item_name'],
            "analysis": result
        })

    return results


# ============================================
# TEST/DEVELOPMENT
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧠 GEMINI 3 FLASH INTELLIGENCE ENGINE TEST")
    print("=" * 70)

    # Test with sample surplus items
    test_items = [
        {"item": "Chicken Biryani", "surplus_kg": 10},
        {"item": "Paneer Tikka", "surplus_kg": 5.5},
        {"item": "Tomato", "surplus_kg": 3}
    ]

    for test in test_items:
        print(f"\n📦 Analyzing: {test['item']} ({test['surplus_kg']}kg)")
        result = get_prescriptive_json(test['item'], test['surplus_kg'])
        print(json.dumps(result, indent=2))
        print("-" * 70)

    print("\n✅ Intelligence Engine Test Complete")
