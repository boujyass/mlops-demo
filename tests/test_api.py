import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_form():
    response = client.get("/")
    assert response.status_code == 200
    # Updated to match the new HTML title
    assert "AI House Value Predictor" in response.text

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

def test_predict_invalid_data():
    """Test API with invalid data"""
    payload = {
        "MedInc": -1,  # Invalid negative income
        "HouseAge": 150,  # Invalid age
        "AveRooms": 0,  # Invalid rooms
        "AveBedrms": 0,  # Invalid bedrooms
        "Population": -100,  # Invalid population
        "AveOccup": 0,  # Invalid occupancy
        "Latitude": 90,  # Invalid latitude for California
        "Longitude": 0  # Invalid longitude for California
    }
    response = client.post("/predict", json=payload)
    # Should still work (model will handle it), but prediction may be unrealistic
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_missing_fields():
    """Test API with missing required fields"""
    payload = {
        "MedInc": 8.3252,
        # Missing other required fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error