import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

df = pd.read_csv("filtered_mapping.csv")
station_codes = list(df["noaa"].astype(str))
station_codes = station_codes[189:]

# print(station_codes)
base_url = "https://www.ncei.noaa.gov/data/global-hourly/access/2018/"
# station_codes = ["010010", "010014", "010020"]  # your 362 codes

# scrape the index to get all available files
resp = requests.get(base_url)
soup = BeautifulSoup(resp.text, "html.parser")

all_files = [a["href"] for a in soup.find_all("a") if a["href"].endswith(".csv")]

# print(len(all_files))

os.makedirs("weather_2018", exist_ok=True)

for code in station_codes:
    matches = [f for f in all_files if f.startswith(code)]
    for fname in matches:
        url = base_url + fname
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(os.path.join("weather_2018", fname), "wb") as f:
                f.write(r.content)
            print(f"Downloaded {fname}")
        else:
            print(f"Failed {fname}")
