import os
import pandas as pd
import numpy as np

def clean_weather_file(infile, outdir):
    df = pd.read_csv(infile)
    df["DATE"] = pd.to_datetime(df["DATE"])

    # --- Handle AT1, AT2 (spread ±12h, then drop daily-only rows) ---
    daily_mask = df["WND"].isna() & (df["AT1"].notna() | df["AT2"].notna())
    for col in ["AT1", "AT2"]:
        events = df.loc[daily_mask & df[col].notna(), ["DATE", col]]
        for _, row in events.iterrows():
            t = row["DATE"]
            mask_window = (df["DATE"] >= t - pd.Timedelta("12h")) & (df["DATE"] <= t + pd.Timedelta("12h"))
            df.loc[mask_window, col] = row[col]
    df = df[~daily_mask]  # drop the daily-only rows

    # --- Continuous vars ---
    cont_interp = ["WND", "CIG", "VIS", "TMP", "DEW"]
    for col in cont_interp:
        df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(limit=3)

    # SLP: interpolate up to 5 missing in a row
    df["SLP"] = pd.to_numeric(df["SLP"], errors="coerce").interpolate(limit=5)

    # OC1: just numeric, no interpolation (keep NaN if missing)
    df["OC1"] = pd.to_numeric(df["OC1"], errors="coerce")

    # Precipitation
    df["AA1"] = pd.to_numeric(df["AA1"], errors="coerce").interpolate(limit=3)  # quantitative
    df["AA2"] = pd.to_numeric(df["AA2"], errors="coerce")  # keep NaN

    # --- Save cleaned file ---
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, os.path.basename(infile))
    df.to_csv(outfile, index=False)
    print(f"Saved cleaned file to {outfile}")


# Batch runner
indir = "weather_2018_cleaned2"
outdir = "weather_2018_final"

for fname in os.listdir(indir):
    if fname.endswith(".csv"):
        infile = os.path.join(indir, fname)
        try:
            clean_weather_file(infile, outdir)
        except Exception as e:
            print(f"Error in {fname}: {e}")
