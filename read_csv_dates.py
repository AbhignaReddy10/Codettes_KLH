import pandas as pd
from io import StringIO

# Your CSV data (using the data you provided)
csv_data = """Date,Item_ID,Item_Name,Category,Subcategory,Unit,Current_Stock,Reorder_Level,Daily_Usage,Lead_Time,Price_per_Unit,Supplier_Name,Seasonal_Factor,Waste_Percentage
2025-06-10,1,Paneer,Veg,Dairy,kg,21.450000000000003,8.12,2.19,1,450,Supplier A,1.11,2.98
2025-06-10,2,Tomato,Veg,Vegetable,kg,12.84,5.34,0.95,4,40,Supplier A,0.81,3.54
2025-06-10,3,Onion,Veg,Vegetable,kg,22.349999999999998,4.49,4.86,4,35,Supplier A,1.23,4.96
2025-06-10,4,Chicken,Non-Veg,Meat,kg,8.36,3.16,3.25,3,250,Supplier C,0.9,3.06
2025-06-10,5,Mutton,Non-Veg,Meat,kg,12.31,6.19,1.81,3,600,Supplier A,1.07,3.09"""

# Read CSV into DataFrame
df = pd.read_csv(StringIO(csv_data))

# Get first 5 raw values from Date column
first_5_dates = df['Date'].head(5)

print("First 5 raw values of the 'Date' column:")
print(first_5_dates)
print("\n" + "="*50)

# Check the type of each value
print("\nData type of each value:")
for i, date_value in enumerate(first_5_dates):
    print(f"Index {i}: {date_value} (Type: {type(date_value).__name__})")

print("\n" + "="*50)
print(f"\nColumn dtype: {df['Date'].dtype}")