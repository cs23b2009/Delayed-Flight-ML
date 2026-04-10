import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from src.utils.logger import setup_logger
from src.config import settings

logger = setup_logger("data_pipeline", settings.LOGS_DIR / "pipeline.log")

class WeatherProcessor:
    """Handles the cleaning, interpolation, and merging of weather data."""
    
    COLUMNS_TO_KEEP = [
        "DATE", "WND", "CIG", "VIS", "TMP", "DEW", "SLP",
        "AA1", "AA2", "AT1", "AT2", "AU1", "AU2",
        "AW1", "AW2", "GD1", "GD2", "OC1"
    ]

    def __init__(self, mapping_path: Optional[Path] = None):
        self.mapping = {}
        if mapping_path and mapping_path.exists():
            mapping_df = pd.read_csv(mapping_path)
            self.mapping = dict(zip(
                mapping_df["noaa"].astype(str).str.zfill(6), 
                mapping_df["iata"]
            ))
            logger.info(f"Loaded mapping with {len(self.mapping)} entries.")

    def parse_component(self, val: any, idx: int, divisor: float = 1.0, missing_vals: List[str] = None) -> float:
        """Parses a specific component from a comma-separated weather value."""
        if pd.isna(val):
            return np.nan
        try:
            parts = str(val).split(",")
            if idx >= len(parts):
                return np.nan
            part = parts[idx].strip()
            if missing_vals and part in missing_vals:
                return np.nan
            return float(part) / divisor
        except (ValueError, IndexError):
            return np.nan

    def clean_raw_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial cleaning and parsing of complex weather columns."""
        cleaned = pd.DataFrame()
        cleaned["DATE"] = pd.to_datetime(df["DATE"])

        # Parsing logic based on ISD format
        cleaned["WND_SPEED"] = df["WND"].apply(lambda x: self.parse_component(x, 3, 10, ["9999", "999.9"]))
        cleaned["CEILING_HEIGHT"] = df["CIG"].apply(lambda x: self.parse_component(x, 0, 1, ["99999"]))
        cleaned["VISIBILITY"] = df["VIS"].apply(lambda x: self.parse_component(x, 0, 1, ["999999"]))
        cleaned["TEMPERATURE"] = df["TMP"].apply(lambda x: self.parse_component(x, 0, 10, ["+9999", "+999.9"]))
        cleaned["DEW_POINT"] = df["DEW"].apply(lambda x: self.parse_component(x, 0, 10, ["+9999", "+999.9"]))
        cleaned["SEA_LEVEL_PRESSURE"] = df["SLP"].apply(lambda x: self.parse_component(x, 0, 10, ["99999", "9999.9"]))
        
        # Precipitation and other indicators
        cleaned["PRECIP_LIQUID"] = df["AA1"].apply(lambda x: self.parse_component(x, 1, 10, ["9999"]))
        
        # Transfer raw indicators for further processing
        for col in ["AT1", "AT2", "WND"]: # Keeping WND for mask check later
            if col in df.columns:
                cleaned[col] = df[col]
        
        return cleaned

    def interpolate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolates missing values to ensure continuous data streams."""
        # Handle weather events (spread ±12h)
        daily_mask = df["WND"].isna() & (df.get("AT1", pd.Series()).notna() | df.get("AT2", pd.Series()).notna())
        for col in ["AT1", "AT2"]:
            if col not in df.columns: continue
            events = df.loc[daily_mask & df[col].notna(), ["DATE", col]]
            for _, row in events.iterrows():
                t = row["DATE"]
                mask_window = (df["DATE"] >= t - pd.Timedelta("12h")) & (df["DATE"] <= t + pd.Timedelta("12h"))
                df.loc[mask_window, col] = row[col]
        
        # Drop rows that were only used for daily event spreading
        df = df[~daily_mask].copy()

        # Interpolation for continuous variables
        cols_to_interp = ["WND_SPEED", "CEILING_HEIGHT", "VISIBILITY", "TEMPERATURE", "DEW_POINT"]
        for col in cols_to_interp:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(limit=3)
        
        if "SEA_LEVEL_PRESSURE" in df.columns:
            df["SEA_LEVEL_PRESSURE"] = pd.to_numeric(df["SEA_LEVEL_PRESSURE"], errors="coerce").interpolate(limit=5)

        return df

    def process_file(self, file_path: Path, output_dir: Path):
        """Processes a single weather station file."""
        try:
            logger.info(f"Processing {file_path.name}...")
            # Detect available columns
            available_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
            cols_to_use = [c for c in self.COLUMNS_TO_KEEP if c in available_cols]
            
            df = pd.read_csv(file_path, usecols=cols_to_use)
            df_cleaned = self.clean_raw_weather(df)
            df_final = self.interpolate_data(df_cleaned)

            # Determine output filename (rename to IATA if mapping exists)
            noaa_code = file_path.name[:6]
            iata_code = self.mapping.get(noaa_code)
            
            output_name = f"{iata_code}.csv" if iata_code else file_path.name
            output_path = output_dir / output_name
            
            df_final.to_csv(output_path, index=False)
            logger.info(f"Successfully processed and saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")

    def batch_process(self, input_dir: Path, output_dir: Path):
        """Processes all CSV files in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        files = list(input_dir.glob("*.csv"))
        logger.info(f"Found {len(files)} files in {input_dir}")
        
        for f in files:
            self.process_file(f, output_dir)
        
        logger.info("Batch processing complete.")
