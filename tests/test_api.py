import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_form():
    response = client.get("/")
    assert response.status_code == 200
    assert "California House Value Prediction" in response.text

def test_predict_valid():
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322,
        "AveOccup": 2.5556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)

def test_predict_ui_valid():
    data = {
        "MedInc": "8.3252",
        "HouseAge": "41",
        "AveRooms": "6.9841",
        "AveBedrms": "1.0238",
        "Population": "322",
        "AveOccup": "2.5556",
        "Latitude": "37.88",
        "Longitude": "-122.23"
    }
    response = client.post("/predict-ui", data=data)
    assert response.status_code == 200
    assert "Predicted House Value" in response.text
