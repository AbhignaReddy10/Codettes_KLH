from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from intelligence import get_prescriptive_json

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Smart Food Supply API Running"}

@app.get("/predict-waste")
def predict_waste(stock: int, demand: int):
    if stock > demand + 40:
        return {"risk": "High Waste Risk"}
    elif stock > demand:
        return {"risk": "Medium Waste Risk"}
    else:
        return {"risk": "Low Waste Risk"}

@app.get("/")
def home():
    return {"status": "AI System Online"}

# This is the part Member 3 will use
@app.post("/calculate_impact")
def calculate(item: str, weight: float):
    # This runs your Biryani code and gives the result back to the website
    result = get_prescriptive_json(item, weight)
    return result
    
@app.get("/check-inventory")
def check_inventory(stock: int):
    if stock < 40:
        return {"status": "Low Stock - Supplier Notified"}
    return {"status": "Stock Level Safe"}

@app.get("/match-ngo")
def match_ngo(surplus: int):
    if surplus > 50:
        return {"ngo": "Sunshine Orphanage"}

    return {"ngo": "No Distribution Needed"}
