import pandas as pd
import warnings
import os
import sys
from prophet import Prophet
from datetime import timedelta
from typing import List, Dict, Optional
import json

warnings.filterwarnings('ignore')

# ============================================
# Stock-Out Detector with Safety Buffer
# ============================================

def detect_stockout_risk(predicted_demand: float, current_stock: float, safety_buffer_percent: float = 10.0) -> Dict:
    """
    Detect stock-out risk with safety buffer.
    
    Logic: If (Predicted Demand * (1 + safety_buffer_percent/100)) > Current Stock,
           generate a RESTOCK_ALERT.
    
    Parameters:
    -----------
    predicted_demand : float
        The forecasted daily usage
    current_stock : float
        The current stock level
    safety_buffer_percent : float
        Safety buffer percentage (default: 10%)
    
    Returns:
    --------
    Dict containing:
        - has_stockout_risk: bool indicating if there's a stock-out risk
        - alert_level: 'NONE', 'ALERT', or 'CRITICAL'
        - demand_with_buffer: Demand including safety buffer
        - buffer_amount: The safety buffer amount in units
        - units_needed: How many units to order to meet demand + buffer
        - message: Human-readable alert message
    """
    
    # Calculate demand with safety buffer
    buffer_amount = predicted_demand * (safety_buffer_percent / 100)
    demand_with_buffer = predicted_demand + buffer_amount
    
    # Calculate units needed
    units_needed = max(0, demand_with_buffer - current_stock)
    
    # Determine alert level
    if current_stock >= demand_with_buffer:
        has_stockout_risk = False
        alert_level = 'NONE'
        message = f"Stock level sufficient. Current: {current_stock:.2f}, Required: {demand_with_buffer:.2f}"
    elif current_stock >= predicted_demand:
        # Stock covers demand but not the safety buffer
        has_stockout_risk = True
        alert_level = 'ALERT'
        message = (
            f"RESTOCK_ALERT: Stock may fall below safety threshold. "
            f"Current: {current_stock:.2f}, Required (with {safety_buffer_percent}% buffer): {demand_with_buffer:.2f}, "
            f"Need to order: {units_needed:.2f} units"
        )
    else:
        # Stock doesn't even cover the predicted demand
        has_stockout_risk = True
        alert_level = 'CRITICAL'
        message = (
            f"CRITICAL_RESTOCK_ALERT: Stock insufficient for predicted demand! "
            f"Current: {current_stock:.2f}, Predicted demand: {predicted_demand:.2f}, "
            f"With safety buffer: {demand_with_buffer:.2f}, "
            f"Need to order: {units_needed:.2f} units IMMEDIATELY"
        )
    
    return {
        'has_stockout_risk': has_stockout_risk,
        'alert_level': alert_level,
        'predicted_demand': round(predicted_demand, 2),
        'demand_with_buffer': round(demand_with_buffer, 2),
        'buffer_amount': round(buffer_amount, 2),
        'buffer_percent': safety_buffer_percent,
        'current_stock': round(current_stock, 2),
        'units_needed': round(units_needed, 2),
        'message': message
    }


# ============================================
# Forecast Function
# ============================================

def predict_next_day_usage(df, item_name):
    """
    Predict the daily usage for an item for the next day after the last date in the dataset.
    """
    item_df = df[df['Item_Name'] == item_name].copy()
    
    if len(item_df) == 0:
        return None
    
    prophet_df = item_df[['Date', 'Daily_Usage']].copy()
    prophet_df.rename(columns={'Date': 'ds', 'Daily_Usage': 'y'}, inplace=True)
    
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df['y'] = pd.to_numeric(prophet_df['y'])
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(prophet_df)
    except Exception as e:
        return None
    
    last_date = prophet_df['ds'].max()
    next_date = last_date + timedelta(days=1)
    
    future_df = pd.DataFrame({'ds': [next_date]})
    forecast = model.predict(future_df)
    
    predicted_value = forecast['yhat'].values[0]
    upper_bound = forecast['yhat_upper'].values[0]
    lower_bound = forecast['yhat_lower'].values[0]
    
    result = {
        'item_name': item_name,
        'next_date': str(next_date.date()),
        'predicted_value': round(predicted_value, 2),
        'upper_bound': round(upper_bound, 2),
        'lower_bound': round(lower_bound, 2),
        'confidence_interval_width': round((upper_bound - lower_bound) / 2, 2),
        'model_details': {
            'records_used': len(prophet_df),
            'date_range': f"{prophet_df['ds'].min().date()} to {last_date.date()}",
            'mean_daily_usage': round(prophet_df['y'].mean(), 2),
            'std_dev_daily_usage': round(prophet_df['y'].std(), 2),
            'min_daily_usage': round(prophet_df['y'].min(), 2),
            'max_daily_usage': round(prophet_df['y'].max(), 2)
        }
    }
    
    return result


# ============================================
# Inventory Risk Assessment Function
# ============================================

def check_inventory_risk(forecast_results, current_stock_df, safety_buffer_percent: float = 10.0):
    """
    Assess inventory risk based on forecast results and current stock levels.
    
    Now includes stock-out detector with safety buffer.
    
    Returns a list of action plans (API-friendly - no print statements)
    """
    
    if not forecast_results:
        return None
    
    # Get the latest stock data for each item
    latest_stock = current_stock_df.sort_values('Date').groupby('Item_Name').last().reset_index()
    
    action_plans = []
    
    for forecast in forecast_results:
        item_name = forecast['item_name']
        predicted_value = forecast['predicted_value']
        upper_bound = forecast['upper_bound']
        lower_bound = forecast['lower_bound']
        
        # Find current stock for this item
        stock_record = latest_stock[latest_stock['Item_Name'] == item_name]
        
        if stock_record.empty:
            continue
        
        current_stock = stock_record['Current_Stock'].values[0]
        
        # ============================================
        # NEW: Stock-Out Detection with Safety Buffer
        # ============================================
        stockout_detection = detect_stockout_risk(
            predicted_demand=predicted_value,
            current_stock=current_stock,
            safety_buffer_percent=safety_buffer_percent
        )
        
        # Calculate shortfall/surplus
        shortfall_critical = max(0, predicted_value - current_stock)
        shortfall_warning = max(0, upper_bound - current_stock)
        surplus = max(0, current_stock - upper_bound)
        
        # Determine Risk Level (now informed by stock-out detection)
        if stockout_detection['alert_level'] == 'CRITICAL':
            risk_level = 'Critical'
        elif stockout_detection['alert_level'] == 'ALERT':
            risk_level = 'Warning'
        else:
            risk_level = 'None'
        
        # Determine Stock Status (now informed by stock-out detection)
        if stockout_detection['has_stockout_risk']:
            status = 'RESTOCK'
            shortfall = stockout_detection['units_needed']
        else:
            status = 'OK'
            shortfall = 0
        
        # Calculate Waste Risk
        waste_ratio = current_stock / predicted_value if predicted_value > 0 else 0
        
        if waste_ratio > 2.0:
            waste_risk = 'High'
        elif waste_ratio > 1.3:
            waste_risk = 'Moderate'
        else:
            waste_risk = 'Low'
        
        # Calculate Risk Score (0-100)
        risk_score = 0
        
        if current_stock < lower_bound:
            risk_score += 50
        elif current_stock < predicted_value:
            risk_score += 35
        elif current_stock < upper_bound:
            risk_score += 20
        else:
            risk_score += 0
        
        if waste_ratio > 2.0:
            risk_score += 40
        elif waste_ratio > 1.5:
            risk_score += 25
        elif waste_ratio > 1.3:
            risk_score += 15
        else:
            risk_score += 0
        
        risk_score = min(100, risk_score)
        
        # Generate Recommended Action
        if stockout_detection['alert_level'] == 'CRITICAL':
            recommended_action = (
                f"🚨 CRITICAL_RESTOCK_ALERT: {stockout_detection['units_needed']:.2f} units needed IMMEDIATELY. "
                f"Current stock ({current_stock:.2f}) is insufficient for predicted demand ({predicted_value:.2f}) "
                f"plus 10% safety buffer ({stockout_detection['buffer_amount']:.2f})."
            )
        elif stockout_detection['alert_level'] == 'ALERT':
            recommended_action = (
                f"⚠️  RESTOCK_ALERT: Order {stockout_detection['units_needed']:.2f} units to maintain safety buffer. "
                f"Current stock covers demand but falls short of safety requirement."
            )
        elif waste_risk == 'High':
            recommended_action = (
                f"Review stock management. Consider reducing order quantities. "
                f"Current stock ({current_stock:.2f}) far exceeds predicted usage ({predicted_value:.2f})."
            )
        elif waste_risk == 'Moderate':
            recommended_action = (
                f"Monitor usage patterns. Current stock is higher than typical usage. "
                f"Adjust reorder quantities if possible."
            )
        else:
            recommended_action = "Continue current inventory management practices."
        
        # Create Action Plan Dictionary
        action_plan = {
            'item_name': item_name,
            'current_stock': round(current_stock, 2),
            'predicted_value': predicted_value,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'status': status,
            'risk_level': risk_level,
            'waste_risk': waste_risk,
            'shortfall': round(shortfall, 2) if shortfall > 0 else 0,
            'surplus': round(surplus, 2) if surplus > 0 else 0,
            'waste_ratio': round(waste_ratio, 2),
            'recommended_action': recommended_action,
            'risk_score': risk_score,
            # NEW: Stock-out detection details
            'stockout_detection': {
                'has_stockout_risk': stockout_detection['has_stockout_risk'],
                'alert_level': stockout_detection['alert_level'],
                'predicted_demand': stockout_detection['predicted_demand'],
                'demand_with_buffer': stockout_detection['demand_with_buffer'],
                'buffer_amount': stockout_detection['buffer_amount'],
                'buffer_percent': stockout_detection['buffer_percent'],
                'units_needed': stockout_detection['units_needed'],
                'alert_message': stockout_detection['message']
            },
            'analysis_metrics': {
                'stock_out_risk_points': min(50, risk_score),
                'waste_risk_points': max(0, risk_score - 50),
                'forecast_confidence': 'High' if forecast['confidence_interval_width'] < 10 else 'Medium' if forecast['confidence_interval_width'] < 15 else 'Low'
            }
        }
        
        action_plans.append(action_plan)
    
    return action_plans


# ============================================
# Enhanced Data Loading Function
# ============================================

def load_inventory_data(filename='cleaned_restaurant_data.csv'):
    """
    Load inventory data from CSV file with comprehensive path search.
    """
    
    # Paths to search
    search_paths = []
    
    # 1. Current working directory
    current_dir = os.path.join(os.getcwd(), filename)
    search_paths.append(('Current Working Directory', current_dir))
    
    # 2. Script directory
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    search_paths.append(('Script Directory', script_dir))
    
    # 3. Parent directory of script
    parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', filename)
    parent_dir = os.path.abspath(parent_dir)
    search_paths.append(('Parent Directory', parent_dir))
    
    # 4. Two levels up from script
    grandparent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', filename)
    grandparent_dir = os.path.abspath(grandparent_dir)
    search_paths.append(('Grandparent Directory', grandparent_dir))
    
    # Try each path
    for location_name, filepath in search_paths:
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                return df
            except Exception as e:
                continue
    
    # If not found, raise an error
    raise FileNotFoundError(
        f"'{filename}' not found in any of the searched locations:\n" +
        "\n".join([f"  - {loc}: {path}" for loc, path in search_paths])
    )


# ============================================
# Main Assessment Function (API-Ready)
# ============================================

def run_full_assessment(items_to_forecast: Optional[List[str]] = None, safety_buffer_percent: float = 10.0) -> Dict:
    """
    Run the complete inventory risk assessment with stock-out detection.
    
    This is the main orchestration function that can be called from FastAPI.
    
    Parameters:
    -----------
    items_to_forecast : Optional[List[str]]
        List of item names to forecast. If None, uses default items.
    safety_buffer_percent : float
        Safety buffer percentage (default: 10%)
    
    Returns:
    --------
    Dict containing:
        - status: 'success' or 'error'
        - message: Description of what happened
        - data: Assessment results (if successful)
        - error_details: Error information (if failed)
    """
    
    try:
        # Set default items if not provided
        if items_to_forecast is None:
            items_to_forecast = ['Paneer', 'Chicken', 'Tomato', 'Milk', 'Rice']
        
        # Load data
        try:
            df = load_inventory_data('data/cleaned_restaurant_data.csv')
        except FileNotFoundError as e:
            return {
                'status': 'error',
                'message': 'Failed to load inventory data',
                'error_details': str(e)
            }
        
        # Prepare data
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Generate forecasts
        forecast_results = []
        forecasting_errors = []
        
        for item_name in items_to_forecast:
            result = predict_next_day_usage(df, item_name)
            if result:
                forecast_results.append(result)
            else:
                forecasting_errors.append(item_name)
        
        if not forecast_results:
            return {
                'status': 'error',
                'message': f'Failed to generate forecasts for any items. Tried: {items_to_forecast}',
                'error_details': f'Failed items: {forecasting_errors}'
            }
        
        # Perform risk assessment with stock-out detection
        action_plans = check_inventory_risk(forecast_results, df, safety_buffer_percent=safety_buffer_percent)
        
        if not action_plans:
            return {
                'status': 'error',
                'message': 'Failed to generate action plans',
                'error_details': 'Risk assessment returned no results'
            }
        
        # Generate summary statistics
        critical_alerts = [p for p in action_plans if p['stockout_detection']['alert_level'] == 'CRITICAL']
        restock_alerts = [p for p in action_plans if p['stockout_detection']['alert_level'] == 'ALERT']
        
        summary_stats = {
            'total_items_assessed': len(action_plans),
            'items_requiring_restock': len([p for p in action_plans if p['status'] == 'RESTOCK']),
            'critical_stockout_alerts': len(critical_alerts),
            'restock_alerts': len(restock_alerts),
            'critical_risk_items': len([p for p in action_plans if p['risk_level'] == 'Critical']),
            'warning_risk_items': len([p for p in action_plans if p['risk_level'] == 'Warning']),
            'high_waste_risk_items': len([p for p in action_plans if p['waste_risk'] == 'High']),
            'average_risk_score': round(sum(p['risk_score'] for p in action_plans) / len(action_plans), 2),
            'total_shortfall': round(sum(p['shortfall'] for p in action_plans), 2),
            'safety_buffer_percent': safety_buffer_percent,
            'data_date_range': {
                'start_date': str(df['Date'].min().date()),
                'end_date': str(df['Date'].max().date())
            }
        }
        
        return {
            'status': 'success',
            'message': 'Assessment completed successfully',
            'data': {
                'action_plans': action_plans,
                'summary_statistics': summary_stats,
                'forecasting_errors': forecasting_errors if forecasting_errors else None
            }
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'message': 'An unexpected error occurred during assessment',
            'error_details': str(e)
        }


# ============================================
# Utility Functions for Export
# ============================================

def export_results_to_csv(action_plans: List[Dict], filename: str = 'inventory_risk_assessment.csv') -> Dict:
    """
    Export action plans to CSV file.
    """
    
    try:
        export_df = pd.DataFrame([
            {
                'Item_Name': p['item_name'],
                'Current_Stock': p['current_stock'],
                'Predicted_Usage': p['predicted_value'],
                'Upper_Bound': p['upper_bound'],
                'Lower_Bound': p['lower_bound'],
                'Status': p['status'],
                'Risk_Level': p['risk_level'],
                'Waste_Risk': p['waste_risk'],
                'Shortfall': p['shortfall'],
                'Surplus': p['surplus'],
                'Risk_Score': p['risk_score'],
                'Stockout_Alert_Level': p['stockout_detection']['alert_level'],
                'Units_Needed': p['stockout_detection']['units_needed'],
                'Recommended_Action': p['recommended_action']
            }
            for p in action_plans
        ])
        
        filepath = os.path.abspath(f'data/{filename}')
        export_df.to_csv(filepath, index=False)
        
        return {
            'status': 'success',
            'message': f'Results exported successfully',
            'filepath': filepath
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': 'Failed to export results',
            'error_details': str(e)
        }


# ============================================
# Main Execution (for testing)
# ============================================

if __name__ == "__main__":
    print("\n" + "="*90)
    print("INVENTORY RISK ASSESSMENT SYSTEM WITH STOCK-OUT DETECTION")
    print("="*90)
    
    # Run the full assessment
    result = run_full_assessment()
    
    # Print results
    print("\n" + "="*90)
    print("ASSESSMENT RESULT")
    print("="*90)
    print(f"\nStatus: {result['status'].upper()}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'success':
        data = result['data']
        stats = data['summary_statistics']
        
        print("\n" + "="*90)
        print("SUMMARY STATISTICS")
        print("="*90)
        print(f"\nTotal Items Assessed: {stats['total_items_assessed']}")
        print(f"Items Requiring Restock: {stats['items_requiring_restock']}")
        print(f"🚨 Critical Stockout Alerts: {stats['critical_stockout_alerts']}")
        print(f"⚠️  Restock Alerts: {stats['restock_alerts']}")
        print(f"Items with Critical Risk: {stats['critical_risk_items']}")
        print(f"Items with Warning Risk: {stats['warning_risk_items']}")
        print(f"Items with High Waste Risk: {stats['high_waste_risk_items']}")
        print(f"Average Risk Score: {stats['average_risk_score']}/100")
        print(f"Total Shortfall: {stats['total_shortfall']:.2f} units")
        print(f"Safety Buffer: {stats['safety_buffer_percent']}%")
        print(f"Data Range: {stats['data_date_range']['start_date']} to {stats['data_date_range']['end_date']}")
        
        print("\n" + "="*90)
        print("ACTION PLANS WITH STOCK-OUT DETECTION")
        print("="*90)
        
        for i, plan in enumerate(data['action_plans'], 1):
            stockout = plan['stockout_detection']
            print(f"\n[{i}] {plan['item_name']}")
            print(f"{'─'*90}")
            print(f"  Current Stock:                {plan['current_stock']:.2f} units")
            print(f"  Predicted Daily Usage:        {plan['predicted_value']:.2f} units")
            print(f"  Predicted Demand + Buffer:    {stockout['demand_with_buffer']:.2f} units ({stockout['buffer_percent']}% safety margin)")
            print(f"  Safety Buffer Amount:         {stockout['buffer_amount']:.2f} units")
            print(f"\n  Status:                       {plan['status']}")
            print(f"  Stockout Alert Level:         {stockout['alert_level']}")
            print(f"  Risk Level:                   {plan['risk_level']}")
            print(f"  Risk Score:                   {plan['risk_score']}/100")
            print(f"  Units Needed to Order:        {stockout['units_needed']:.2f} units")
            print(f"\n  Alert Message:                {stockout['alert_message']}")
            print(f"  Recommended Action:           {plan['recommended_action']}")
        
        # Export results
        print("\n" + "="*90)
        export_result = export_results_to_csv(data['action_plans'])
        print(f"Export Status: {export_result['status'].upper()}")
        if export_result['status'] == 'success':
            print(f"File saved to: {export_result['filepath']}")
    else:
        print(f"Error Details: {result.get('error_details', 'No additional details')}")
    
    print("\n" + "="*90)
