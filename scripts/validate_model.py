import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.models import FlightModel
from src.config import settings

def test_model_load():
    try:
        model = FlightModel(
            model_path=settings.CATBOOST_MODEL_PATH,
            template_path=settings.PROCESSED_DATA_DIR / "final_merged_1000_rows.csv"
        )
        if model.model is not None:
            print("✅ Model loaded successfully!")
            print(f"Features loaded: {len(model.feature_names)}")
            
            # Test prediction with dummy data
            dummy_input = {
                "Origin": "ATL",
                "Dest": "TPA",
                "CRSDepTime": 1950,
                "CRSArrTime": 2118,
                "Distance": 406.0,
                "Month": 12,
                "DayOfWeek": 1,
                "IATA_Code_Marketing_Airline": "DL"
            }
            result = model.predict(dummy_input)
            print(f"✅ Prediction test successful: {result}")
        else:
            print("❌ Model failed to load.")
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")

if __name__ == "__main__":
    test_model_load()
