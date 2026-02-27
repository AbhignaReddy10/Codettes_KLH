from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from clean_restaurant_inventory import clean_inventory
from read_csv_dates import clean_sales
from inventory_risk_check import analyze_inventory
from intelligence import generate_ai_plan

load_dotenv()

app = FastAPI(title="AI Food Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze(
    sales_file: UploadFile = File(...),
    inventory_file: UploadFile = File(...)
):

    try:
        sales_df = clean_sales(sales_file.file)
        inventory_df = clean_inventory(inventory_file.file)

        restock_alerts, surplus_items = analyze_inventory(inventory_df, sales_df)

        ai_result = generate_ai_plan(restock_alerts, surplus_items)

        return {
            "status": "success",
            "restock_alerts": restock_alerts,
            "surplus_items": surplus_items,
            "ai_plan": ai_result["ai_plan"],
            "confidence_score": ai_result["confidence_score"]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
