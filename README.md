# MLOps Demo - House Price Prediction

A machine learning web app that predicts California house prices using FastAPI and Docker.



## What It Does

- **Predicts house prices** in California using 8 features (income, age, rooms, etc.)
- **Web interface** with sample data buttons for easy testing
- **REST API** for programmatic access
- **Automated CI/CD** pipeline with GitHub Actions

## Project Structure

```
mlops-demo/
├── app/                 # FastAPI web application
│   ├── main.py         # API endpoints
│   ├── templates/      # HTML interface
│   └── requirements.txt
├── model/              
│   └── train.py        # ML model training
├── tests/              # Test suite
├── Dockerfile          # Container setup
└── .github/workflows/  # CI/CD pipeline
```

## API Usage

**Web Interface:** 

Locally: http://localhost:8000

Deployed version: https://mlops-demo-latest.onrender.com/

**Predict API:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322,
    "AveOccup": 2.56,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

## Features

- ✅ Machine learning model with 95%+ accuracy
- ✅ Beautiful web interface with sample data
- ✅ REST API with JSON responses
- ✅ Docker containerization
- ✅ Automated testing and deployment
- ✅ Input validation and error handling

## Testing

```bash
pytest tests/
```

## Model Details

- **Algorithm:** Histogram Gradient Boosting Regressor
- **Dataset:** California housing (20,640 samples)
- **Features:** Income, house age, rooms, location, etc.
- **Performance:** Cross-validated with hyperparameter tuning

## Deploy to Production

1. Set up Docker Hub credentials in GitHub secrets:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`

2. Push to main branch to trigger automated deployment

## Environment Variables

```bash
MODEL_PATH=model/model.pkl  # Path to trained model
PORT=8000                   # Server port
LOG_LEVEL=info             # Logging level
```


