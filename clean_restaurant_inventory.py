import pandas as pd
import numpy as np
from datetime import datetime

# Read the CSV file
df = pd.read_csv('datasets/restaurant_inventory_100days.csv')

print("Original DataFrame shape:", df.shape)
print("\nFirst few rows before cleaning:")
print(df.head())

# ============================================
# 1. Convert 'Date' column to datetime objects
# ============================================
print("\n" + "="*60)
print("Step 1: Converting Date column to datetime")
print("="*60)

# Check if dates are strings or Excel serial numbers
print(f"Original Date dtype: {df['Date'].dtype}")
print(f"Sample values: {df['Date'].head()}")

# Try to convert as string first (YYYY-MM-DD format)
try:
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    print("✓ Successfully converted Date as datetime strings")
except Exception as e:
    print(f"String conversion failed: {e}")
    # If string conversion fails, try Excel serial number conversion
    try:
        df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1899-12-30')
        print("✓ Successfully converted Date as Excel serial numbers")
    except Exception as e2:
        print(f"Excel conversion also failed: {e2}")

print(f"Converted Date dtype: {df['Date'].dtype}")
print(f"Sample converted dates: {df['Date'].head()}")

# ============================================
# 2. Clean 'Item_Name' column (remove extra spaces)
# ============================================
print("\n" + "="*60)
print("Step 2: Cleaning Item_Name column")
print("="*60)

# Check for extra spaces before cleaning
print(f"Sample Item_Name values before cleaning:")
print(df['Item_Name'].head(10).tolist())

# Remove leading/trailing spaces and replace multiple spaces with single space
df['Item_Name'] = df['Item_Name'].str.strip()
df['Item_Name'] = df['Item_Name'].str.replace(r'\s+', ' ', regex=True)

print(f"\nSample Item_Name values after cleaning:")
print(df['Item_Name'].head(10).tolist())

# ============================================
# 3. Fill missing values
# ============================================
print("\n" + "="*60)
print("Step 3: Handling missing values")
print("="*60)

# Check for missing values
print("Missing values before filling:")
print(df.isnull().sum())

# Fill 'Waste_Percentage' with median
if df['Waste_Percentage'].isnull().any():
    median_waste = df['Waste_Percentage'].median()
    df['Waste_Percentage'].fillna(median_waste, inplace=True)
    print(f"\n✓ Filled Waste_Percentage missing values with median: {median_waste}")
else:
    print("\n✓ No missing values in Waste_Percentage")

# Fill 'Current_Stock' with 0
if df['Current_Stock'].isnull().any():
    df['Current_Stock'].fillna(0, inplace=True)
    print("✓ Filled Current_Stock missing values with 0")
else:
    print("✓ No missing values in Current_Stock")

print("\nMissing values after filling:")
print(df.isnull().sum())

# ============================================
# 4. Add 'Is_Weekend' boolean column
# ============================================
print("\n" + "="*60)
print("Step 4: Adding Is_Weekend column")
print("="*60)

# 0=Monday, 6=Sunday
# Weekend = Saturday (5) or Sunday (6)
df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6])

print("Sample Is_Weekend values:")
print(df[['Date', 'Is_Weekend']].head(15))
print(f"✓ Is_Weekend column added (dtype: {df['Is_Weekend'].dtype})")

# ============================================
# 5. Add 'Day_of_Week' string column
# ============================================
print("\n" + "="*60)
print("Step 5: Adding Day_of_Week column")
print("="*60)

# Map day numbers to day names
day_names = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

df['Day_of_Week'] = df['Date'].dt.dayofweek.map(day_names)

print("Sample Day_of_Week values:")
print(df[['Date', 'Day_of_Week', 'Is_Weekend']].head(15))
print(f"✓ Day_of_Week column added (dtype: {df['Day_of_Week'].dtype})")

# ============================================
# 6. Export to cleaned CSV
# ============================================
print("\n" + "="*60)
print("Step 6: Exporting cleaned data")
print("="*60)

output_filename = 'datasets/cleaned_restaurant_data.csv'
df.to_csv(output_filename, index=False)
print(f"✓ Successfully exported cleaned data to '{output_filename}'")

# ============================================
# Summary
# ============================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total rows processed: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nFinal DataFrame info:")
print(df.info())
print(f"\nFirst 10 rows of cleaned data:")
print(df.head(10))
print(f"\nLast 10 rows of cleaned data:")
print(df.tail(10))

# Display statistics on new columns
print(f"\n--- New Columns Statistics ---")
print(f"\nIs_Weekend value counts:")
print(df['Is_Weekend'].value_counts())
print(f"\nDay_of_Week value counts:")
print(df['Day_of_Week'].value_counts().sort_index())