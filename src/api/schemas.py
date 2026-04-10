from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class FlightPredictionRequest(BaseModel):
    """Schema for flight cancellation prediction request."""
    Origin: str = Field(..., example="ATL", description="Origin airport IATA code")
    Dest: str = Field(..., example="TPA", description="Destination airport IATA code")
    CRSDepTime: int = Field(..., example=1950, description="Scheduled departure time (HHMM)")
    CRSArrTime: int = Field(..., example=2118, description="Scheduled arrival time (HHMM)")
    Distance: float = Field(..., example=406.0, description="Flight distance in miles")
    Month: int = Field(..., ge=1, le=12, example=12)
    DayOfWeek: int = Field(..., ge=1, le=7, example=1)
    IATA_Code_Marketing_Airline: str = Field(..., example="DL", description="Marketing airline IATA code")

    class Config:
        schema_extra = {
            "example": {
                "Origin": "ATL",
                "Dest": "TPA",
                "CRSDepTime": 1950,
                "CRSArrTime": 2118,
                "Distance": 406.0,
                "Month": 12,
                "DayOfWeek": 1,
                "IATA_Code_Marketing_Airline": "DL"
            }
        }

class PredictionResponse(BaseModel):
    """Schema for flight cancellation prediction response."""
    will_cancel: bool
    cancel_probability: float
    threshold: float = 0.5
    input_received: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    model_loaded: bool
    version: str
