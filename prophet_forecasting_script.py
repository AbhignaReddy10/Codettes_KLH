from pathlib import Path
import pandas as pd
import warnings
from prophet import Prophet
from datetime import timedelta

warnings.filterwarnings('ignore')

# ============================================
# Forecasting Function using Prophet
# ============================================

def predict_next_day_usage(df, item_name):
    """
    Predict the daily usage for an item for the next day after the last date in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Date', 'Item_Name', and 'Daily_Usage' columns
    item_name : str
        Name of the item to forecast
    
    Returns:
    --------
    dict : Dictionary containing:
        - predicted_value: Point forecast for next day
        - upper_bound: Upper confidence interval (95%)
        - lower_bound: Lower confidence interval (95%)
        - next_date: The forecasted date
        - item_name: Name of the item
        - model_details: Additional info about the model
    """
    
    print(f"\n{'='*70}")
    print(f"Forecasting Daily Usage for: {item_name}")
    print(f"{'='*70}")
    
    # Step 1: Filter data for the given item
    print(f"\n[Step 1] Filtering data for '{item_name}'...")
    item_df = df[df['Item_Name'] == item_name].copy()
    
    if len(item_df) == 0:
        print(f"❌ Error: No data found for item '{item_name}'")
        return None
    
    print(f"✓ Found {len(item_df)} records for '{item_name}'")
    print(f"  Date range: {item_df['Date'].min()} to {item_df['Date'].max()}")
    
    # Step 2: Prepare data for Prophet (rename columns)
    print(f"\n[Step 2] Preparing data for Prophet...")
    prophet_df = item_df[['Date', 'Daily_Usage']].copy()
    prophet_df.rename(columns={'Date': 'ds', 'Daily_Usage': 'y'}, inplace=True)
    
    # Ensure ds is datetime and y is numeric
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df['y'] = pd.to_numeric(prophet_df['y'])
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    print(f"✓ Data prepared for Prophet")
    print(f"  Shape: {prophet_df.shape}")
    print(f"  Daily Usage Statistics:")
    print(f"    Mean: {prophet_df['y'].mean():.2f}")
    print(f"    Std Dev: {prophet_df['y'].std():.2f}")
    print(f"    Min: {prophet_df['y'].min():.2f}")
    print(f"    Max: {prophet_df['y'].max():.2f}")
    
    # Step 3: Initialize and fit Prophet model
    print(f"\n[Step 3] Fitting Prophet model...")
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(prophet_df)
        print(f"✓ Prophet model fitted successfully")
    except Exception as e:
        print(f"❌ Error fitting model: {e}")
        return None
    
    # Step 4: Create future dataframe for next day prediction
    print(f"\n[Step 4] Creating forecast for next day...")
    last_date = prophet_df['ds'].max()
    next_date = last_date + timedelta(days=1)
    
    future_df = pd.DataFrame({'ds': [next_date]})
    
    print(f"  Last date in dataset: {last_date.date()}")
    print(f"  Forecasting for: {next_date.date()}")
    
    # Step 5: Make prediction
    print(f"\n[Step 5] Generating forecast...")
    forecast = model.predict(future_df)
    
    # Extract forecast values
    predicted_value = forecast['yhat'].values[0]
    upper_bound = forecast['yhat_upper'].values[0]
    lower_bound = forecast['yhat_lower'].values[0]
    
    print(f"✓ Forecast generated")
    print(f"  Predicted Daily Usage: {predicted_value:.2f}")
    print(f"  Upper Bound (95% CI): {upper_bound:.2f}")
    print(f"  Lower Bound (95% CI): {lower_bound:.2f}")
    print(f"  Confidence Interval Width: ±{(upper_bound - lower_bound) / 2:.2f}")
    
    # Step 6: Prepare result dictionary
    result = {
        'item_name': item_name,
        'next_date': next_date.date(),
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
# Main Script Execution
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RESTAURANT INVENTORY USAGE FORECASTING")
    print("Using Facebook Prophet Library")
    print("="*70)
    
    # Load the cleaned data
    print("\n[Loading Data] Reading 'datasets/cleaned_restaurant_data.csv'...")
    try:
        df = pd.read_csv('datasets/cleaned_restaurant_data.csv')
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("❌ Error: 'datasets/cleaned_restaurant_data.csv' not found")
        print("   Please ensure the file exists in the current directory")
        exit(1)
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Display basic info
    print(f"\n[Data Summary]")
    print(f"  Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Unique Items: {df['Item_Name'].nunique()}")
    print(f"  Items: {sorted(df['Item_Name'].unique())}")
    
    # ============================================
    # Test forecasting for multiple items
    # ============================================
    
    items_to_forecast = ['Paneer', 'Chicken']
    results = []
    
    print("\n" + "="*70)
    print("FORECASTING RESULTS")
    print("="*70)
    
    for item_name in items_to_forecast:
        result = predict_next_day_usage(df, item_name)
        if result:
            results.append(result)
    
    # ============================================
    # Print Summary Results
    # ============================================
    
    print("\n" + "="*70)
    print("FORECAST SUMMARY TABLE")
    print("="*70)
    
    if results:
        summary_df = pd.DataFrame([
            {
                'Item': r['item_name'],
                'Next Date': r['next_date'],
                'Predicted Usage': r['predicted_value'],
                'Lower Bound': r['lower_bound'],
                'Upper Bound': r['upper_bound'],
                'Confidence Width': r['confidence_interval_width']
            }
            for r in results
        ])
        
        print("\n" + summary_df.to_string(index=False))
        
        # ============================================
        # Print Detailed Results
        # ============================================
        
        print("\n" + "="*70)
        print("DETAILED FORECAST RESULTS")
        print("="*70)
        
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}] {result['item_name']}")
            print(f"{'-'*70}")
            print(f"  Forecasted Date: {result['next_date']}")
            print(f"  Predicted Daily Usage: {result['predicted_value']} units")
            print(f"  Confidence Interval (95%):")
            print(f"    Lower Bound: {result['lower_bound']} units")
            print(f"    Upper Bound: {result['upper_bound']} units")
            print(f"    Range: [{result['lower_bound']}, {result['upper_bound']}]")
            print(f"  Forecast Uncertainty (±): {result['confidence_interval_width']} units")
            
            print(f"\n  Model Training Details:")
            details = result['model_details']
            print(f"    Total Records Used: {details['records_used']}")
            print(f"    Training Date Range: {details['date_range']}")
            print(f"    Mean Daily Usage: {details['mean_daily_usage']} units")
            print(f"    Std Dev: {details['std_dev_daily_usage']} units")
            print(f"    Min Daily Usage: {details['min_daily_usage']} units")
            print(f"    Max Daily Usage: {details['max_daily_usage']} units")
        
        # ============================================
        # Interpretation Guide
        # ============================================
        
        print("\n" + "="*70)
        print("INTERPRETATION GUIDE")
        print("="*70)
        print("""
The forecast provides three key metrics:

1. PREDICTED VALUE
   - The most likely daily usage for the next day
   - Use this for your baseline inventory planning

2. CONFIDENCE INTERVAL (95%)
   - Lower Bound: 95% probability usage will be >= this value
   - Upper Bound: 95% probability usage will be <= this value
   - Interpretation: In 95 out of 100 similar days, usage will fall
     within this range

3. FORECAST UNCERTAINTY
   - Represented by the width of the confidence interval
   - Wider interval = Higher uncertainty in the forecast
   - Narrower interval = More confident in the prediction

USAGE RECOMMENDATIONS:
- Stock inventory at the predicted value for normal operations
- Maintain safety stock at the lower bound level
- Plan for peak demand using the upper bound value
        """)
    
    print("\n" + "="*70)
    print("FORECASTING COMPLETE")
    print("="*70 + "\n")