import json
import pandas as pd

with open("results.json", "r") as f:
    results = json.load(f)

df = pd.DataFrame.from_dict(results, orient="index")

print(df)

df.to_csv("results_summary.csv")