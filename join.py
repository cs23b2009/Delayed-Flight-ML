import pandas as pd

# Load the two sheets
iata_icao = pd.read_csv("iata-icao.csv")   # columns: IATA, ICAO
usaf_icao = pd.read_csv("icao-noaa.csv")   # columns: USAF, ICAO

# Merge on ICAO
merged = pd.merge(iata_icao, usaf_icao, on="icao", how="left")

# Now merged has columns: IATA, ICAO, USAF
print(merged.head())

# If you just want IATA → USAF mapping
# mapping = merged[['IATA', 'NOAA']]
# print(mapping)
merged.to_csv("iata-noaa.csv")
