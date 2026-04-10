# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier
import pandas as pd
import os

# ---------- 1. Pydantic model: what the USER sends ----------

class FlightSimpleInput(BaseModel):
    Origin: str = Field(example="ATL")
    Dest: str = Field(example="TPA")
    CRSDepTime: int = Field(example=1950, description="HHMM, e.g. 1950 = 19:50")
    CRSArrTime: int = Field(example=2118, description="HHMM, e.g. 2118 = 21:18")
    Distance: float = Field(example=406)
    Month: int = Field(ge=1, le=12, example=12)
    DayOfWeek: int = Field(ge=1, le=7, example=1)
    IATA_Code_Marketing_Airline: str = Field(example="DL")


# ---------- 2. Create app ----------

app = FastAPI(
    title="Flight Cancellation API",
    description="Predicts if a flight will be cancelled using a CatBoost model.",
    version="1.0.0",
)

# Paths relative to this file
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "cancellation_model.cbm")
TEMPLATE_DATA_PATH = os.path.join(BASE_DIR, "final_merged_1000_rows.csv")

# Globals populated at startup
model: CatBoostClassifier | None = None
COLUMN_ORDER: list[str] = []
DEFAULT_ROW: dict[str, object] = {}


# ---------- 3. Startup: load model + template data ----------
# ---------------------- Paste this replacement into app/main.py ----------------------

from typing import Dict, Set

ENCODERS: Dict[str, Dict[str, int]] = {}
MODEL_FEATURE_NAMES: list[str] | None = None
NUMERIC_COLS: Set[str] = set()

@app.on_event("startup")
def startup_event():
    """
    Demo startup:
    - load model,
    - read the template CSV and use its FIRST ROW as DEFAULT_ROW,
    - infer numeric columns (from CSV dtypes),
    - build encoders for Origin/Dest/IATA,
    - read model feature names (if available),
    - ensure defaults for encoded columns.
    """
    global model, COLUMN_ORDER, DEFAULT_ROW, ENCODERS, MODEL_FEATURE_NAMES, NUMERIC_COLS

    # 1) Load CatBoost model
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    # 2) Read CSV - read a small sample plus full header for dtypes
    # keep_default_na=False to avoid turning empty strings into NaN
    df_sample = pd.read_csv(TEMPLATE_DATA_PATH, dtype=str, keep_default_na=False, nrows=50)
    df_full_head = pd.read_csv(TEMPLATE_DATA_PATH, nrows=5)  # let pandas infer numeric dtypes for detection

    if df_sample.shape[0] == 0:
        raise RuntimeError("Template CSV is empty or not found at: " + TEMPLATE_DATA_PATH)

    # Save CSV column order
    COLUMN_ORDER = df_sample.columns.tolist()

    # Determine numeric-like columns using df_full_head's dtypes
    NUMERIC_COLS = set([c for c, dt in df_full_head.dtypes.items() if pd.api.types.is_numeric_dtype(dt)])

    # Use the FIRST ROW of df_sample as DEFAULT_ROW, but coerce numeric-like columns to numbers
    first = df_sample.iloc[0].to_dict()
    DEFAULT_ROW = {}
    for k, v in first.items():
        if v == "" or v is None:
            DEFAULT_ROW[k] = v
            continue
        if k in NUMERIC_COLS:
            # try numeric conversion robustly
            try:
                num = float(v)
                if abs(num - int(num)) < 1e-9:
                    DEFAULT_ROW[k] = int(num)
                else:
                    DEFAULT_ROW[k] = float(num)
            except Exception:
                # fallback to original string if conversion fails
                DEFAULT_ROW[k] = v
        else:
            DEFAULT_ROW[k] = v

    # Drop obvious index-like keys from DEFAULT_ROW to avoid shifting columns
    for bad in ["Unnamed: 0", "__index_level_0__"]:
        DEFAULT_ROW.pop(bad, None)

    # 3) Build simple encoders (factorize mapping order) for the three categories
    cat_cols_to_encode = ["Origin", "Dest", "IATA_Code_Marketing_Airline"]
    ENCODERS = {}
    df_for_enc = pd.read_csv(TEMPLATE_DATA_PATH, dtype=str, keep_default_na=False)
    for c in cat_cols_to_encode:
        if c in df_for_enc.columns:
            vals = df_for_enc[c].astype(str).tolist()
            mapping = {}
            for v in vals:
                if v not in mapping:
                    mapping[v] = len(mapping)
            ENCODERS[c] = mapping
        else:
            ENCODERS[c] = {}

    # 4) Model feature names if available
    try:
        if hasattr(model, "feature_names_"):
            MODEL_FEATURE_NAMES = list(model.feature_names_)
        elif hasattr(model, "feature_names"):
            MODEL_FEATURE_NAMES = list(model.feature_names)
        else:
            MODEL_FEATURE_NAMES = None
    except Exception:
        MODEL_FEATURE_NAMES = None

    # 5) Ensure encoded defaults exist
    for c in cat_cols_to_encode:
        enc_name = f"{c}_encoded"
        default_cat = str(DEFAULT_ROW.get(c, ""))
        DEFAULT_ROW[enc_name] = int(ENCODERS.get(c, {}).get(default_cat, 0))

    print("✅ Demo startup complete. Using first CSV row as DEFAULT_ROW.")
    print("CSV columns:", len(COLUMN_ORDER))
    print("Detected numeric columns count:", len(NUMERIC_COLS))
    if MODEL_FEATURE_NAMES:
        print("Model feature count from model:", len(MODEL_FEATURE_NAMES))
    else:
        print("Model feature names not found in CatBoost object; will fallback to CSV intersection.")


def build_full_feature_row(user_input: FlightSimpleInput) -> pd.DataFrame:
    """
    Build full feature row with careful dtype handling similar to notebook:
      - Start from DEFAULT_ROW (first CSV row),
      - override with incoming user_input,
      - coerce numeric-like columns to numeric (NUMERIC_COLS),
      - ensure categorical fields are strings,
      - create encoded *_encoded fields using ENCODERS,
      - fill any missing model-expected columns from DEFAULT_ROW or safe defaults,
      - reorder to MODEL_FEATURE_NAMES if available,
      - robust final coercion to ensure CatBoost-safe dtypes.
    """
    # 1) start with defaults
    row = dict(DEFAULT_ROW)

    # 2) override with user-provided values
    for k, v in user_input.dict().items():
        row[k] = v

    # 3) Coerce types: numeric columns -> numeric; categorical -> string
    for col in list(row.keys()):
        val = row.get(col)
        if val is None or val == "":
            # will fill later from DEFAULT_ROW if needed
            continue
        if col in NUMERIC_COLS:
            # try numeric conversion robustly
            try:
                nv = float(val)
                if abs(nv - int(nv)) < 1e-9:
                    row[col] = int(nv)
                else:
                    row[col] = float(nv)
            except Exception:
                # if conversion fails, fall back to default or 0
                if col in DEFAULT_ROW:
                    row[col] = DEFAULT_ROW[col]
                else:
                    row[col] = 0
        else:
            # categorical-like -> string
            row[col] = str(val)

    # 4) Ensure categorical names are strings (explicit)
    for c in ["Origin", "Dest", "IATA_Code_Marketing_Airline"]:
        if c in row:
            row[c] = "" if row[c] is None else str(row[c])

    # 5) Create encoded columns using ENCODERS; unseen -> safe code = len(mapping)
    for c, mapping in ENCODERS.items():
        enc_name = f"{c}_encoded"
        cat_val = str(row.get(c, "")) if row.get(c, "") is not None else ""
        if mapping and cat_val in mapping:
            row[enc_name] = int(mapping[cat_val])
        else:
            row[enc_name] = int(len(mapping) if mapping else 0)

    # 6) Remove obviously irrelevant fields that might shift ordering
    for rem in ["Tail_Number", "Dep_DateTime", "Arr_DateTime", "Unnamed: 0", "__index_level_0__"]:
        row.pop(rem, None)

    # 7) Construct DataFrame
    df = pd.DataFrame([row])

    # 8) Align to model feature names if available, else use CSV intersection (stable)
    if MODEL_FEATURE_NAMES:
        # add missing features using DEFAULT_ROW or safe defaults
        for feat in MODEL_FEATURE_NAMES:
            if feat not in df.columns:
                if feat in DEFAULT_ROW:
                    df[feat] = DEFAULT_ROW[feat]
                else:
                    # safe default heuristics
                    if feat.endswith("_encoded") or feat.startswith(("o_", "d_", "FlightID")):
                        df[feat] = 0
                    else:
                        # if expected numeric, use 0 else empty string
                        df[feat] = 0 if feat in NUMERIC_COLS else ""
        # final reorder
        df = df[MODEL_FEATURE_NAMES]
    else:
        # fallback: keep CSV order intersection to be stable
        cols = [c for c in COLUMN_ORDER if c in df.columns]
        df = df[cols]

    # --- Robust final coercion to avoid CatBoost categorical-type errors ---
    # Try to get cat feature indices from the CatBoost model (if available)
    cat_feature_names = set()
    try:
        # CatBoost has get_cat_feature_indices() in some versions
        if hasattr(model, "get_cat_feature_indices"):
            idxs = model.get_cat_feature_indices()
            if MODEL_FEATURE_NAMES and idxs:
                for i in idxs:
                    if 0 <= i < len(MODEL_FEATURE_NAMES):
                        cat_feature_names.add(MODEL_FEATURE_NAMES[i])
        # fallback: some wrappers expose 'cat_features' param
        elif hasattr(model, "get_param"):
            try:
                cf = model.get_param("cat_features") if callable(model.get_param) else None
                if cf:
                    for name in cf:
                        cat_feature_names.add(str(name))
            except Exception:
                pass
    except Exception:
        # ignore if model introspection fails
        pass

    # Heuristic: columns that are integer-valued floats should be integers (fixes 0.0 -> 0)
    for col in df.columns:
        # 1) If model says this is categorical, ensure it's int or str
        if col in cat_feature_names:
            # if all values are integral floats, convert to integer
            if pd.api.types.is_float_dtype(df[col]):
                # check integrality
                if df[col].dropna().apply(lambda x: abs(x - round(x)) < 1e-9).all():
                    df[col] = df[col].fillna(0).astype(int)
                else:
                    # not integral floats -> convert to strings (safe categorical form)
                    df[col] = df[col].astype(str).fillna("")
            else:
                # if already int or object, ensure proper type
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = df[col].fillna(0).astype(int)
                else:
                    df[col] = df[col].astype(str).fillna("")
            continue

        # 2) For other columns: if float but integral-valued, cast to int (avoid accidental float categories)
        if pd.api.types.is_float_dtype(df[col]):
            if df[col].dropna().apply(lambda x: abs(x - round(x)) < 1e-9).all():
                df[col] = df[col].fillna(0).astype(int)
            else:
                # keep as float with NaNs filled if needed
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # 3) Ensure our explicit encoded columns are integer
        if col.endswith("_encoded"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # 4) Ensure main string categories are strings
        if col in ["Origin", "Dest", "IATA_Code_Marketing_Airline"]:
            df[col] = df[col].astype(str).fillna("")

    return df


# ---------------------- end paste ----------------------


# # ---------- 5. Prediction endpoint ----------

# @app.post("/predict")
# def predict_cancellation(features: FlightSimpleInput):
#     """
#     Takes simple flight fields, fills in the rest with defaults,
#     and returns cancellation probability.
#     """
#     full_row = build_full_feature_row(features)

#     # CatBoost predict_proba -> [ [prob_class0, prob_class1] ]
#     proba = model.predict_proba(full_row)[0, 1]  # class "1" = cancelled

#     prediction = proba >= 0.5

#     return {
#         "input_used": features.dict(),
#         "cancel_probability": float(proba),
#         "will_cancel": bool(prediction),
#         "threshold": 0.5,
#     }

from fastapi import HTTPException

@app.post("/predict")
def predict_cancellation(features: FlightSimpleInput):
    """
    Diagnostic wrapper: builds the full row, attempts prediction, and on error returns
    useful debugging info (columns, shapes, dtypes, sample values, and any model info).
    """
    try:
        full_row = build_full_feature_row(features)
    except Exception as e:
        # error building the row (likely a missing column name)
        raise HTTPException(status_code=500, detail={
            "error": "failed_building_full_row",
            "message": str(e),
        })

    # Basic diagnostics we'll return if predict fails
    diag = {
        "full_row_shape": full_row.shape,
        "full_row_columns": list(full_row.columns),
        "full_row_dtypes": {c: str(full_row[c].dtype) for c in full_row.columns},
        "full_row_sample": full_row.iloc[0].to_dict(),
        "num_model_features_expected": None,
        "model_feature_names": None,
    }

    # try to get model's feature names/expected size if available
    try:
        # CatBoost sometimes exposes feature_names_ or feature_names
        if hasattr(model, "feature_names_"):
            diag["model_feature_names"] = list(model.feature_names_)
            diag["num_model_features_expected"] = len(model.feature_names_)
        elif hasattr(model, "feature_names"):
            diag["model_feature_names"] = list(model.feature_names)
            diag["num_model_features_expected"] = len(model.feature_names)
        elif hasattr(model, "get_feature_names"):
            try:
                diag["model_feature_names"] = list(model.get_feature_names())
                diag["num_model_features_expected"] = len(diag["model_feature_names"])
            except Exception:
                diag["model_feature_names"] = "get_feature_names() failed"
        else:
            # fallback: if model was trained via sklearn wrapper XGBClassifier it may have n_features_in_
            if hasattr(model, "n_features_in_"):
                diag["num_model_features_expected"] = int(model.n_features_in_)
    except Exception as e:
        diag["model_info_error"] = str(e)

    try:
        proba = model.predict_proba(full_row)[0, 1]
        prediction = bool(proba >= 0.5)
        return {
            "input_used": features.dict(),
            "cancel_probability": float(proba),
            "will_cancel": prediction,
            "threshold": 0.5,
            "diagnostics": diag,
        }
    except Exception as e:
        # print to console too
        print("=== Prediction error ===")
        print(str(e))
        import traceback
        traceback.print_exc()

        # return 500 with diagnostic information
        raise HTTPException(status_code=500, detail={
            "error": "prediction_failed",
            "message": str(e),
            "diagnostics": diag
        })
