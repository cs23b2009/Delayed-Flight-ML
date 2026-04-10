import pandas as pd
from catboost import CatBoostClassifier
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger("model_loader", settings.LOGS_DIR / "model.log")

class FlightModel:
    """Handles loading and inference for the flight cancellation model."""
    
    def __init__(self, model_path: Path, template_path: Optional[Path] = None):
        self.model_path = model_path
        self.template_path = template_path
        self.model = None
        self.default_row = {}
        self.column_order = []
        self.encoders = {}
        self.numeric_cols = set()
        self.feature_names = []
        
        self.load_model()
        if template_path:
            self.load_template_data()

    def load_model(self):
        """Loads the CatBoost model from disk."""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return
        
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(str(self.model_path))
            # Extract feature names if available
            if hasattr(self.model, "feature_names_"):
                self.feature_names = list(self.model.feature_names_)
            logger.info("Successfully loaded CatBoost model.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")

    def load_template_data(self):
        """Infers defaults and encoders from template CSV."""
        if not self.template_path or not self.template_path.exists():
            logger.warning("Template path not found. Skipping template loading.")
            return

        try:
            # Read a small sample for dtypes and a full set for encoders
            df_full = pd.read_csv(self.template_path, dtype=str, keep_default_na=False)
            df_sample = df_full.head(50)
            
            self.column_order = df_full.columns.tolist()
            
            # Infer numeric columns
            df_numeric_infer = pd.read_csv(self.template_path, nrows=5)
            self.numeric_cols = set([c for c, dt in df_numeric_infer.dtypes.items() if pd.api.types.is_numeric_dtype(dt)])

            # Build encoders for categorical columns
            cat_cols = ["Origin", "Dest", "IATA_Code_Marketing_Airline"]
            for c in cat_cols:
                if c in df_full.columns:
                    unique_vals = df_full[c].unique()
                    self.encoders[c] = {v: i for i, v in enumerate(unique_vals)}
            
            # Create a default row from the first row of sample
            first_row = df_sample.iloc[0].to_dict()
            self.default_row = {}
            for k, v in first_row.items():
                if k in self.numeric_cols and v != "":
                    try:
                        self.default_row[k] = float(v)
                    except:
                        self.default_row[k] = v
                else:
                    self.default_row[k] = v
            
            # Cleanup unwanted columns
            for bad in ["Unnamed: 0", "__index_level_0__"]:
                self.default_row.pop(bad, None)
                
            logger.info("Template data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading template data: {str(e)}")

    def preprocess(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocesses input dictionary into a model-ready DataFrame."""
        row = dict(self.default_row)
        row.update(input_data)

        # Handle encoding
        for c, mapping in self.encoders.items():
            enc_name = f"{c}_encoded"
            cat_val = str(row.get(c, ""))
            row[enc_name] = mapping.get(cat_val, len(mapping))

        df = pd.DataFrame([row])

        # Align columns
        if self.feature_names:
            for feat in self.feature_names:
                if feat not in df.columns:
                    df[feat] = self.default_row.get(feat, 0 if feat in self.numeric_cols else "")
            df = df[self.feature_names]
        
        # Final type coercion to ensure CatBoost compatibility
        for col in df.columns:
            if col.endswith("_encoded") or col in self.numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                if abs(df[col].iloc[0] - round(df[col].iloc[0])) < 1e-9:
                    df[col] = df[col].astype(int)
            else:
                df[col] = df[col].astype(str)
        
        return df

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs prediction on input data."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        try:
            processed_df = self.preprocess(input_data)
            proba = self.model.predict_proba(processed_df)[0, 1]
            return {
                "probability": float(proba),
                "prediction": bool(proba >= 0.5)
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise e
