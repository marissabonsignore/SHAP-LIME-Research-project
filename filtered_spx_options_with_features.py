import pandas as pd
import numpy as np

INPUT_FILE = "data/filtered_spx_options_with_greeks.csv"
OUTPUT_FILE = "data/filtered_spx_options_with_features.csv"

print("=== ADDING CORE FEATURES ===")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)
print(f"Loaded rows: {len(df)}")

# --------------------------------------------------
# ENSURE NUMERIC TYPES
# --------------------------------------------------
df["S"] = pd.to_numeric(df["S"], errors="coerce")
df["K"] = pd.to_numeric(df["K"], errors="coerce")
df["T"] = pd.to_numeric(df["T"], errors="coerce")
df["Implied Volatility"] = pd.to_numeric(df["Implied Volatility"], errors="coerce")

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

# Log-moneyness: ln(K / S)
df["Log_Moneyness"] = np.log(df["K"] / df["S"])

# Time-to-maturity already exists as T
# Included explicitly for clarity / modeling
df["Time_to_Maturity"] = df["T"]

# Implied volatility already exists
df["IV"] = df["Implied Volatility"]

# --------------------------------------------------
# FILTER INVALID VALUES (OPTIONAL BUT RECOMMENDED)
# --------------------------------------------------
valid_mask = (
    np.isfinite(df["Log_Moneyness"]) &
    (df["Time_to_Maturity"] > 0) &
    (df["IV"] > 0)
)

df = df.loc[valid_mask].copy()

print("Remaining rows after feature filtering:", len(df))

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("SUCCESS")
print(f"Saved feature-enhanced dataset to: {OUTPUT_FILE}")
print("=== DONE ===")