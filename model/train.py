import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Pipeline: scaling + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', HistGradientBoostingRegressor(random_state=42))
])

# Hyperparameter grid for tuning
param_distributions = {
    'model__max_iter': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__min_samples_leaf': [20, 50, 100],
}

# Randomized search with 5-fold CV
search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

print("Best params:", search.best_params_)

best_model = search.best_estimator_

# Evaluate on test set
preds = best_model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Test MSE: {mse:.4f}")

# Save best model pipeline
joblib.dump(best_model, "model/model.pkl")
