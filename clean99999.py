import os
import pandas as pd
import numpy as np

# Input and output folders
input_folder = "weather_2018_cleaned"
output_folder = "weather_2018_cleaned2"
os.makedirs(output_folder, exist_ok=True)

# Columns you are keeping
columns = [
    "DATE", "WND", "CIG", "VIS", "TMP", "DEW", "SLP",
    "AA1", "AA2",
    "AT1", "AT2",
    "AU1", "AU2",
    "AW1", "AW2",
    "GD1", "GD2",
    "OC1"
]

# Parsing helpers
def parse_split(val, idx, divisor=1, missing_vals=None):
    if pd.isna(val):
        return np.nan
    try:
        part = val.split(",")[idx].strip()
        if part in missing_vals:
            return np.nan
        return float(part) / divisor
    except Exception:
        return np.nan

def parse_code(val, idx):
    if pd.isna(val):
        return np.nan
    try:
        return val.split(",")[idx].strip()
    except Exception:
        return np.nan


for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    
    fpath = os.path.join(input_folder, fname)
    df = pd.read_csv(fpath, usecols=columns)

    cleaned = pd.DataFrame()
    cleaned["DATE"] = df["DATE"]

    cleaned["WND"] = df["WND"].apply(lambda x: parse_split(x, 3, 10, {"9999", "999.9"}))
    cleaned["CIG"] = df["CIG"].apply(lambda x: parse_split(x, 0, 1, {"99999"}))
    cleaned["VIS"] = df["VIS"].apply(lambda x: parse_split(x, 0, 1, {"999999"}))
    cleaned["TMP"] = df["TMP"].apply(lambda x: parse_split(x, 0, 10, {"+9999", "+999.9"}))
    cleaned["DEW"] = df["DEW"].apply(lambda x: parse_split(x, 0, 10, {"+9999", "+999.9"}))
    cleaned["SLP"] = df["SLP"].apply(lambda x: parse_split(x, 0, 10, {"99999", "9999.9"}))

    cleaned["AA1"] = df["AA1"].apply(lambda x: parse_split(x, 1, 10, {"9999"}))
    cleaned["AA2"] = df["AA2"].apply(lambda x: parse_split(x, 1, 10, {"9999"}))

    cleaned["AT1"] = df["AT1"].apply(lambda x: parse_code(x, 1))
    cleaned["AT2"] = df["AT2"].apply(lambda x: parse_code(x, 1))

    cleaned["AU1"] = df["AU1"].apply(lambda x: parse_code(x, 0))
    cleaned["AU2"] = df["AU2"].apply(lambda x: parse_code(x, 0))

    cleaned["AW1"] = df["AW1"].apply(lambda x: parse_code(x, 0))
    cleaned["AW2"] = df["AW2"].apply(lambda x: parse_code(x, 0))

    cleaned["GD1"] = df["GD1"].apply(lambda x: parse_code(x, 0))
    cleaned["GD2"] = df["GD2"].apply(lambda x: parse_code(x, 0))

    cleaned["OC1"] = df["OC1"].apply(lambda x: parse_split(x, 0, 10, {"9999", "999.9"}))

    outpath = os.path.join(output_folder, fname)
    cleaned.to_csv(outpath, index=False)
    print(f"{fname} cleaned and saved.")

print("All files processed.")
