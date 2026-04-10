import os
import pandas as pd

# Load mapping
mapping_df = pd.read_csv("iata-noaa.csv")
# assume columns are ["IATA", "NOAA"], adjust if different
mapping = dict(zip(mapping_df["noaa"].astype(str).str.zfill(6), mapping_df["iata"]))

indir = "weather_2018_final"
outdir = "weather_2018_final_renamed"
os.makedirs(outdir, exist_ok=True)

for fname in os.listdir(indir):
    if fname.endswith(".csv"):
        infile = os.path.join(indir, fname)

        # extract NOAA code from filename (first 6 digits)
        noaa_code = fname[:6]
        if noaa_code in mapping:
            iata_code = mapping[noaa_code]
            new_name = f"{iata_code}.csv"
            outfile = os.path.join(outdir, new_name)
            os.rename(infile, outfile)
            print(f"Renamed {fname} -> {new_name}")
        else:
            print(f"Warning: {fname} not found in mapping")
