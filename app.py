# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib

app = FastAPI(title="Flight Disruption Prediction API")

# Pydantic input (you already had a similar model)
class FlightInput(BaseModel):
    origin: str
    dest: str
    airline: str
    month: int
    day_of_week: int
    sched_dep_hour: int
    distance: float
    temperature_origin: float
    wind_speed_origin: float
    visibility_origin: float
    temperature_dest: float
    wind_speed_dest: float
    visibility_dest: float

# --- Load models & pipeline at startup ---
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
CATBOOST_PATH = os.path.join(MODEL_DIR, "catboost_model.cbm")
PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing.joblib")

model_cancel = None
preprocessor = None

@app.on_event("startup")
def load_models():
    global model_cancel, preprocessor
    # Load CatBoost model (classifier/regressor depending on what you saved)
    if os.path.exists(CATBOOST_PATH):
        try:
            # If you trained a classifier:
            model_cancel = CatBoostClassifier()
            model_cancel.load_model(CATBOOST_PATH)

            # If you trained a regressor for delay, load similarly:
            # model_delay = CatBoostRegressor(); model_delay.load_model(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CatBoost model: {e}")
    else:
        # Fall back to None and raise error on request
        model_cancel = None

    # Optional preprocessing pipeline
    if os.path.exists(PIPELINE_PATH):
        preprocessor = joblib.load(PIPELINE_PATH)
    else:
        preprocessor = None

# --- Helper to build feature vector from FlightInput ---
def make_feature_vector(data: FlightInput):
    # Convert input into a 1D array matching training order.
    # **IMPORTANT**: this must match the exact order/encoding used during training.
    feat = [
        data.origin,
        data.dest,
        data.airline,
        data.month,
        data.day_of_week,
        data.sched_dep_hour,
        data.distance,
        data.temperature_origin,
        data.wind_speed_origin,
        data.visibility_origin,
        data.temperature_dest,
        data.wind_speed_dest,
        data.visibility_dest,
    ]
    return feat

@app.post("/predict")
def predict_flight_disruption(data: FlightInput):
    # Ensure model loaded
    if model_cancel is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    # Build feature vector (as DataFrame if your preprocessor expects one)
    raw_feat = make_feature_vector(data)

    # If you used a pipeline that expects a DataFrame, create it the same way as during training.
    # Minimal example: turn into 2D np.array (1 sample)
    X = np.array([raw_feat], dtype=object)

    # If preprocessor exists, transform
    if preprocessor is not None:
        try:
            X_trans = preprocessor.transform(X)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")
    else:
        X_trans = X  # if model expects raw fields in this shape

    # Predict cancellation probability (CatBoostClassifier.predict_proba expects numeric array)
    try:
        proba = model_cancel.predict_proba(X_trans)[0, 1]  # probability of class 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # If you also have a delay model, predict expected delay minutes similarly.
    # For now, return a placeholder for delay if not available.
    return {
        "cancellation_probability": float(round(proba, 4)),
        "expected_delay_minutes": None
    }
