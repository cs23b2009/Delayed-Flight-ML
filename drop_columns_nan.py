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

os.makedirs("weather_2018_cleaned_2", exist_ok=True)

nan_files = [
    "72012063837.csv",
    "72061600207.csv",
    "72213263801.csv",
    "72213653883.csv",
    "72268893034.csv",
    "72306513783.csv",
    "72389403181.csv",
    "72428513812.csv",
    "72503814714.csv",
    "72518699999.csv",
    "72520754735.csv",
    "72586594161.csv",
    "72645704825.csv",
    "72646594890.csv",
    "72676494163.csv",
    "78514011603.csv",
    "91190422552.csv"
]

i=1

for f in nan_files:
    df = pd.read_csv(os.path.join("weather_2018", f), low_memory=False)

        # Add missing columns as empty
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[columns]

    
    new_file = f"{f.split('.')[0]}_1.csv"
    new_path = os.path.join("weather_2018_cleaned_2", new_file)

    df.to_csv(new_path, index=False)

    print(f"{i} done")
    i += 1

    del(df)