import pandas as pd
import numpy as np
from scipy.stats import norm

INPUT_FILE = "data/filtered_spx_options.csv"
OUTPUT_FILE = "data/filtered_spx_options_with_greeks.csv"

RISK_FREE_RATE = 0.03
DEFAULT_T = 30 / 365

print("=== SCRIPT STARTED ===")

df = pd.read_csv(INPUT_FILE)
print(f"Loaded rows: {len(df)}")

df["S"] = pd.to_numeric(df["Mid Price"], errors="coerce")

df["K"] = pd.to_numeric(df["Open Interest"], errors="coerce")

df["T"] = DEFAULT_T
df["Implied Volatility"] = pd.to_numeric(df["Implied Volatility"], errors="coerce")

print("\nNon-null counts:")
print(df[["S", "K", "T", "Implied Volatility"]].notna().sum())

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def delta(S, K, T, r, sigma, flag):
    return norm.cdf(d1(S, K, T, r, sigma)) if flag == "C" else norm.cdf(d1(S, K, T, r, sigma)) - 1

def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    return S * norm.pdf(d1(S, K, T, r, sigma)) * np.sqrt(T)

mask = (
    (df["S"] > 0) &
    (df["K"] > 0) &
    (df["Implied Volatility"] > 0)
)

df.loc[mask, "Delta"] = df.loc[mask].apply(
    lambda x: delta(x["S"], x["K"], x["T"], RISK_FREE_RATE,
                    x["Implied Volatility"], x["Put Call Flag"]),
    axis=1
)

df.loc[mask, "Gamma"] = df.loc[mask].apply(
    lambda x: gamma(x["S"], x["K"], x["T"], RISK_FREE_RATE,
                    x["Implied Volatility"]),
    axis=1
)

df.loc[mask, "Vega"] = df.loc[mask].apply(
    lambda x: vega(x["S"], x["K"], x["T"], RISK_FREE_RATE,
                   x["Implied Volatility"]),
    axis=1
)

df.to_csv(OUTPUT_FILE, index=False)

print("SUCCESS")
print(f"Greek rows populated: {mask.sum()}")
print("=== SCRIPT FINISHED ===")