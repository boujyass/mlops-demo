from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model/model.pkl")

class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                          data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    prediction = model.predict(features)[0]
    return {"predicted_house_value": prediction}
