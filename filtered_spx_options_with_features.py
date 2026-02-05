import pandas as pd
import numpy as np

INPUT_FILE = "data/filtered_spx_options_with_greeks.csv"
OUTPUT_FILE = "data/filtered_spx_options_with_features.csv"

print("=== ADDING CORE FEATURES ===")

df = pd.read_csv(INPUT_FILE)
print(f"Loaded rows: {len(df)}")

df["S"] = pd.to_numeric(df["S"], errors="coerce")
df["K"] = pd.to_numeric(df["K"], errors="coerce")
df["T"] = pd.to_numeric(df["T"], errors="coerce")
df["Implied Volatility"] = pd.to_numeric(df["Implied Volatility"], errors="coerce")

df["Log_Moneyness"] = np.log(df["K"] / df["S"])

df["Time_to_Maturity"] = df["T"]

df["IV"] = df["Implied Volatility"]

valid_mask = (
    np.isfinite(df["Log_Moneyness"]) &
    (df["Time_to_Maturity"] > 0) &
    (df["IV"] > 0)
)

df = df.loc[valid_mask].copy()

print("Remaining rows after feature filtering:", len(df))

df.to_csv(OUTPUT_FILE, index=False)

print("SUCCESS")
print(f"Saved feature-enhanced dataset to: {OUTPUT_FILE}")
print("=== DONE ===")