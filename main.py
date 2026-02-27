from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import json
import os
from typing import Optional, List
import pandas as pd

# ============================================
# MEMBER 1 IMPORTS (Data & Forecasting)
# ============================================
from inventory_stock_risk import (
    run_full_assessment,
    export_results_to_csv,
    load_inventory_data,
    predict_next_day_usage,
    check_inventory_risk,
    detect_stockout_risk
)

# ============================================
# MEMBER 2 IMPORTS (YOUR INTELLIGENCE)
# ============================================
from intelligence import get_prescriptive_json

# ============================================
# FastAPI Setup
# ============================================
app = FastAPI(
    title="Smart Kitchen Waste Management API",
    description="3-Phase Waste Reduction: Predict → Redirect → Re-invent",
    version="1.0.0"
)

# CORS Middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
def home():
    """API Health Check"""
    return {
        "message": "🚀 Smart Kitchen Waste Management API Running",
        "version": "1.0.0",
        "status": "active",
        "members": {
            "member_1": "Data Architect (Forecasting & Risk Assessment with Stock-Out Detection)",
            "member_2": "Intelligence Engine (Gemini AI & NGO Matching)",
            "member_3": "Frontend & Integration"
        },
        "features": [
            "Prophet time-series forecasting",
            "Stock-out risk detection with safety buffer",
            "Waste prediction and NGO matching",
            "Real-time inventory alerts"
        ],
        "endpoints": {
            "assessment": "/run-assessment",
            "surplus_analysis": "/analyze-surplus",
            "health": "/"
        }
    }


# ============================================
# PHASE A: PREDICT (Member 1 Integration)
# ============================================

@app.get("/run-assessment")
def run_full_inventory_assessment(
        items: Optional[str] = None,
        safety_buffer: float = 10.0
):
    """
    🔵 PHASE A: PREDICT - Identifies Future Waste & Stock-Out Risk

    Triggers Member 1's complete inventory risk assessment using:
    - Prophet forecasting for next-day demand
    - Stock-out detection with safety buffer
    - Inventory risk scoring (0-100)
    - Shortfall/Surplus calculation
    - Waste risk categorization

    Query Parameters:
    ----------------
    items : str (optional)
        Comma-separated item names. Default: 'Paneer,Chicken,Tomato,Milk,Rice'
    safety_buffer : float (optional)
        Safety buffer percentage (default: 10%)

    Returns:
    --------
    {
        "status": "success",
        "data": {
            "action_plans": [...],              # Detailed per-item analysis
            "summary_statistics": {...},        # High-level metrics with stockout alerts
            "forecasting_errors": [...]         # Items that failed to forecast
        }
    }
    """
    try:
        # Parse items from query parameter
        items_list = None
        if items:
            items_list = [item.strip() for item in items.split(',')]

        # Run Member 1's full assessment with stock-out detection
        result = run_full_assessment(
            items_to_forecast=items_list,
            safety_buffer_percent=safety_buffer
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Assessment failed: {str(e)}"
        )


@app.get("/forecast/{item_name}")
def get_item_forecast(item_name: str):
    """
    Get next-day forecast for a specific item.

    Uses Prophet time-series forecasting to predict:
    - Daily usage (point estimate)
    - Confidence intervals (95%)
    - Model confidence metrics
    """
    try:
        # Load data
        df = load_inventory_data('data/cleaned_restaurant_data.csv')

        # Generate forecast
        forecast = predict_next_day_usage(df, item_name)

        if forecast is None:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data for item: {item_name}"
            )

        return {
            "status": "success",
            "item": item_name,
            "forecast": forecast
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed: {str(e)}"
        )


@app.get("/stockout-risk/{item_name}")
def get_stockout_risk(
        item_name: str,
        predicted_demand: float,
        current_stock: float,
        safety_buffer: float = 10.0
):
    """
    🚨 Check stock-out risk for a specific item with safety buffer.

    Parameters:
    -----------
    item_name : str
        Name of the item
    predicted_demand : float
        Forecasted daily usage
    current_stock : float
        Current stock level
    safety_buffer : float
        Safety buffer percentage (default: 10%)

    Returns:
    --------
    Stock-out risk assessment with alert level (NONE, ALERT, CRITICAL)
    """
    try:
        stockout_check = detect_stockout_risk(
            predicted_demand=predicted_demand,
            current_stock=current_stock,
            safety_buffer_percent=safety_buffer
        )

        return {
            "status": "success",
            "item": item_name,
            "stockout_analysis": stockout_check
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/restock-alerts")
def get_restock_alerts(safety_buffer: float = 10.0):
    """
    🚨 Get items requiring immediate restock.

    Returns items where:
    - Stock-out risk is detected (ALERT or CRITICAL)
    - Current Stock < Predicted Usage + Safety Buffer
    """
    try:
        result = run_full_assessment(safety_buffer_percent=safety_buffer)

        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail="Assessment failed")

        # Separate by alert level
        critical_items = [
            plan for plan in result['data']['action_plans']
            if plan['stockout_detection']['alert_level'] == 'CRITICAL'
        ]

        alert_items = [
            plan for plan in result['data']['action_plans']
            if plan['stockout_detection']['alert_level'] == 'ALERT'
        ]

        return {
            "status": "success",
            "total_alerts": len(critical_items) + len(alert_items),
            "critical_stockout_alerts": critical_items,
            "restock_alerts": alert_items,
            "priority": "CRITICAL" if critical_items else ("HIGH" if alert_items else "NONE"),
            "safety_buffer_percent": safety_buffer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/high-waste-risk")
def get_high_waste_risk_items():
    """
    🔴 Get items at high waste risk.

    Returns items where:
    - Current Stock > 2x Predicted Usage, OR
    - Waste Risk = "High"
    """
    try:
        result = run_full_assessment()

        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail="Assessment failed")

        # Filter for high waste risk items
        waste_risk_items = [
            plan for plan in result['data']['action_plans']
            if plan['waste_risk'] == 'High'
        ]

        return {
            "status": "success",
            "count": len(waste_risk_items),
            "high_risk_items": waste_risk_items,
            "total_surplus_kg": round(
                sum(plan['surplus'] for plan in waste_risk_items), 2
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# PHASE B: REDIRECT (Member 2 - YOUR Intelligence)
# ============================================

@app.post("/analyze-surplus")
def analyze_surplus_with_ai(
        item_name: str,
        surplus_kg: float,
        reason: Optional[str] = None
):
    """
    🟢 PHASE B: REDIRECT - AI-Powered NGO Matching & Impact Analysis

    Takes surplus food and uses Gemini 3 Flash to:
    1. Analyze WHY surplus happened
    2. Recommend nearest qualified NGO
    3. Calculate impact metrics (CO2, meals, cost)
    4. Generate handling instructions

    Parameters:
    -----------
    item_name : str
        Food item name (e.g., "Chicken Biryani")
    surplus_kg : float
        Surplus quantity in kg
    reason : str (optional)
        Context for surplus (e.g., "Festival cancellation")

    Returns:
    --------
    {
        "item": "Chicken Biryani",
        "surplus_kg": 10.5,
        "ai_analysis": {
            "reasoning": "High demand for festival not realized...",
            "ngo_recommendation": "Smile Foundation (3km, 500+ daily)",
            "impact_metrics": {
                "co2_saved_kg": 26.25,
                "meals_provided": 105,
                "cost_saved_inr": 2625
            },
            "handling_instructions": "Keep at 4°C, consume within 6hrs"
        },
        "status": "success"
    }
    """
    try:
        # Validate input
        if surplus_kg <= 0:
            raise HTTPException(
                status_code=400,
                detail="Surplus must be > 0 kg"
            )

        # Call your Gemini intelligence
        ai_result = get_prescriptive_json(item_name, surplus_kg)

        # Check for errors from Gemini
        if "error" in ai_result:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini API error: {ai_result['error']}"
            )

        # Structure response for frontend
        response = {
            "item": item_name,
            "surplus_kg": surplus_kg,
            "reason_context": reason,
            "ai_analysis": ai_result,
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat()
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Surplus analysis failed: {str(e)}"
        )


@app.post("/process-waste-event")
def process_waste_detection(
        item_name: str,
        waste_kg: float,
        detection_method: str = "manual"
):
    """
    🔄 CLOSED LOOP: When waste is detected, immediately:
    1. Call Member 2's AI to determine NGO match
    2. Log the waste event
    3. Return action (redistribute/donate)

    Detection Methods:
    - "manual": User reported
    - "sensor": Auto-detected by IoT
    - "forecast": Predicted by algorithm
    """
    try:
        # Trigger AI analysis
        ai_result = get_prescriptive_json(item_name, waste_kg)

        return {
            "event_type": "waste_detection",
            "item": item_name,
            "waste_kg": waste_kg,
            "detection_method": detection_method,
            "recommended_action": ai_result.get('ngo_recommendation', 'Manual review'),
            "impact": ai_result.get('impact_metrics', {}),
            "status": "processed",
            "timestamp": pd.Timestamp.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# PHASE C: RE-INVENT (Stock Optimization)
# ============================================

@app.get("/optimization-suggestions")
def get_optimization_suggestions(safety_buffer: float = 10.0):
    """
    🟣 PHASE C: RE-INVENT - Use historical waste to fix future

    Analyzes past surplus/waste and recommends:
    - Order quantity reductions
    - Menu optimization (daily specials for high-risk items)
    - Reorder level adjustments
    - Safety buffer optimization
    """
    try:
        result = run_full_assessment(safety_buffer_percent=safety_buffer)

        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail="Assessment failed")

        action_plans = result['data']['action_plans']

        # Generate optimization suggestions
        suggestions = {
            "high_waste_items": [
                {
                    "item": plan['item_name'],
                    "current_stock": plan['current_stock'],
                    "current_order_qty": plan['current_stock'],
                    "suggested_reduction": round(plan['surplus'] * 0.5, 2),
                    "waste_risk": plan['waste_risk'],
                    "reason": plan['recommended_action'],
                    "expected_savings_inr": round(plan['surplus'] * 100, 2)
                }
                for plan in action_plans if plan['waste_risk'] == 'High'
            ],
            "menu_specials": [
                {
                    "item": plan['item_name'],
                    "suggested_discount": "20-30%",
                    "target_qty_to_sell": round(plan['current_stock'] * 0.8, 2),
                    "current_surplus": plan['surplus'],
                    "ngo_donation_qty": round(plan['surplus'] * 0.5, 2)
                }
                for plan in action_plans if plan['surplus'] > 5
            ],
            "stockout_prevention": [
                {
                    "item": plan['item_name'],
                    "current_stock": plan['current_stock'],
                    "demand_with_buffer": plan['stockout_detection']['demand_with_buffer'],
                    "units_to_order": plan['stockout_detection']['units_needed'],
                    "safety_buffer_percent": safety_buffer,
                    "alert_level": plan['stockout_detection']['alert_level']
                }
                for plan in action_plans
                if plan['stockout_detection']['has_stockout_risk']
            ]
        }

        return {
            "status": "success",
            "optimization_data": suggestions,
            "safety_buffer_percent": safety_buffer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# DATA EXPORT & REPORTING
# ============================================

@app.get("/export-assessment-csv")
def export_assessment_csv(safety_buffer: float = 10.0):
    """
    Export current assessment results to CSV file.
    Downloads as: inventory_risk_assessment.csv
    """
    try:
        result = run_full_assessment(safety_buffer_percent=safety_buffer)

        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail="Assessment failed")

        # Export to CSV
        export_result = export_results_to_csv(result['data']['action_plans'])

        if export_result['status'] != 'success':
            raise HTTPException(
                status_code=500,
                detail=f"Export failed: {export_result['message']}"
            )

        # Return file for download
        return FileResponse(
            path=export_result['filepath'],
            filename='inventory_risk_assessment.csv',
            media_type='text/csv'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard-summary")
def get_dashboard_summary(safety_buffer: float = 10.0):
    """
    📊 Complete dashboard summary for Member 3's frontend.

    Returns all metrics needed for real-time display including:
    - Stockout alerts (critical + warning)
    - Waste risk items
    - Optimization suggestions
    """
    try:
        result = run_full_assessment(safety_buffer_percent=safety_buffer)

        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail="Assessment failed")

        stats = result['data']['summary_statistics']
        plans = result['data']['action_plans']

        return {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "summary": stats,
            "action_items": {
                "critical_stockout_alerts": stats['critical_stockout_alerts'],
                "restock_alerts": stats['restock_alerts'],
                "high_waste_risk": stats['high_waste_risk_items'],
                "critical_risk_items": stats['critical_risk_items']
            },
            "top_risks": sorted(plans, key=lambda x: x['risk_score'], reverse=True)[:5],
            "top_waste_items": sorted(plans, key=lambda x: x['surplus'], reverse=True)[:5],
            "top_stockout_risks": sorted(
                [p for p in plans if p['stockout_detection']['has_stockout_risk']],
                key=lambda x: x['stockout_detection']['units_needed'],
                reverse=True
            )[:5],
            "safety_buffer_percent": safety_buffer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# SIMPLE LEGACY ENDPOINTS (Keep for compatibility)
# ============================================

@app.get("/predict-waste")
def predict_waste(stock: int, demand: int):
    """Simple waste risk prediction (legacy)"""
    if stock > demand + 40:
        return {"risk": "High Waste Risk", "confidence": 0.85}
    elif stock > demand:
        return {"risk": "Medium Waste Risk", "confidence": 0.6}
    else:
        return {"risk": "Low Waste Risk", "confidence": 0.9}


@app.get("/check-inventory")
def check_inventory(stock: int):
    """Check if restock needed (legacy)"""
    return {
        "status": "Low Stock - Supplier Notified" if stock < 40 else "Stock Level Safe",
        "action": "RESTOCK_ALERT" if stock < 40 else "MONITOR"
    }


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail}
    )
