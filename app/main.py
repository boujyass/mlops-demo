from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Check if model file exists
model_path = "model/model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please run 'python model/train.py' first.")

model = joblib.load(model_path)
templates = Jinja2Templates(directory="app/templates")

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES, "prediction": None})

@app.post("/predict")
async def predict_api(data: InputData):
    """API endpoint for JSON predictions"""
    features = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms, 
                         data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request,
    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    AveOccup: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...),
):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(features)[0]
    return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES, "prediction": round(prediction, 3)})