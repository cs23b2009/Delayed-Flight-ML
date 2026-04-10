import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data Subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MAPPING_DATA_DIR = DATA_DIR / "mapping"

# Model Paths
CATBOOST_MODEL_PATH = MODELS_DIR / "cancellation_model.cbm"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgmodel_delay.joblib"

# API Settings
API_TITLE = "Flight Prediction & Analysis API"
API_VERSION = "2.0.0"
API_DESCRIPTION = "Production-ready API for flight disruption and delay prediction."

# Ensure directories exist
for path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MAPPING_DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)
