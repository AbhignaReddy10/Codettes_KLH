def analyze_inventory(inventory_df, sales_df):
    """
    Detect restock risks and surplus items.
    """

    restock_alerts = []
    surplus_items = []

    avg_sales = sales_df.groupby("item")["quantity_sold"].mean().to_dict()

    for _, row in inventory_df.iterrows():
        item = row["item"]
        stock = row["stock"]
        threshold = row["threshold"]

        avg_daily_sale = avg_sales.get(item, 0)

        if stock <= threshold:
            restock_alerts.append({
                "item": item,
                "current_stock": float(stock),
                "threshold": float(threshold),
                "avg_daily_sales": round(avg_daily_sale, 2)
            })

        if avg_daily_sale > 0 and stock > avg_daily_sale * 7:
            surplus_items.append({
                "item": item,
                "current_stock": float(stock),
                "estimated_weekly_need": round(avg_daily_sale * 7, 2)
            })

    return restock_alerts, surplus_items
