from fastapi import FastAPI, HTTPException, Request
from src.api.schemas import FlightPredictionRequest, PredictionResponse, HealthResponse
from src.core.models import FlightModel
from src.config import settings
from src.utils.logger import setup_logger
import time

# Initialize logger
logger = setup_logger("api", settings.LOGS_DIR / "api.log")

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Global model instance
flight_model = None

@app.on_event("startup")
def startup_event():
    """Load the model and dependencies on startup."""
    global flight_model
    logger.info("Starting up API and loading models...")
    try:
        flight_model = FlightModel(
            model_path=settings.CATBOOST_MODEL_PATH,
            template_path=settings.PROCESSED_DATA_DIR / "final_merged_1000_rows.csv"
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Critical error during startup: {str(e)}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log request timing."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Request {request.method} {request.url.path} handled in {duration:.4f}s")
    return response

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Checks if the service is healthy and model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": flight_model.model is not None if flight_model else False,
        "version": settings.API_VERSION
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_cancellation(request: FlightPredictionRequest):
    """Main endpoint for flight cancellation prediction."""
    if not flight_model or flight_model.model is None:
        logger.error("Prediction attempt with model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    
    try:
        # Perform prediction
        result = flight_model.predict(request.dict())
        
        return {
            "will_cancel": result["prediction"],
            "cancel_probability": result["probability"],
            "threshold": 0.5,
            "input_received": request.dict(),
            "metadata": {
                "model_version": "catboost-v1",
                "features_used": len(flight_model.feature_names) if flight_model.feature_names else 0
            }
        }
    except Exception as e:
        logger.error(f"Prediction failed for request {request.dict()}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
def get_info():
    """Returns metadata about the available models."""
    return {
        "title": settings.API_TITLE,
        "description": settings.API_DESCRIPTION,
        "features_count": len(flight_model.feature_names) if flight_model and flight_model.feature_names else "N/A",
        "numeric_columns": list(flight_model.numeric_cols) if flight_model else [],
        "categorical_columns": list(flight_model.encoders.keys()) if flight_model else []
    }
