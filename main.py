from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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