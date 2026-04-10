import os
import pandas as pd

columns = [
    "DATE", "WND", "CIG", "VIS", "TMP", "DEW", "SLP",
    "AA1", "AA2",
    "AT1", "AT2",
    "AU1", "AU2",
    "AW1", "AW2",
    "GD1", "GD2",
    "OC1"
]

os.makedirs("weather_2018_cleaned", exist_ok=True)

for f in os.listdir("weather_2018"):
    df = pd.read_csv(os.path.join("weather_2018", f), low_memory=False)

    try:
        df = df[columns]

        
        new_file = f"{f.split('.')[0]}_1.csv"
        new_path = os.path.join("weather_2018_cleaned", new_file)

        df.to_csv(new_path, index=False)

        # print(f"{f} done")

    except KeyError as e:
        print(f"{f} is erroneous. error: {e}")
    
    finally:
        del(df)