import pandas as pd
import os

INPUT_FILE = "merged_spx_options.csv"
OUTPUT_FILE = "data/filtered_spx_options.csv"

CHUNKSIZE = 250_000 

DROP_COLS = [
    "Block Volume",
    "Number of Price Moves",
    "Turnover",
    "VWAP Volume",
    "Net Asset Value",
    "Reference Company"
]

first_chunk = True

print("Starting filtering process...")

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE):

    chunk = chunk[
        (chunk["Bid"] > 0) &
        (chunk["Ask"] > chunk["Bid"])
    ]

    chunk = chunk[
        (chunk["Volume"] > 0) |
        (chunk["Open Interest"] > 0)
    ]

    chunk = chunk[chunk["Put Call Flag"].isin(["C", "P"])]

    chunk = chunk[chunk["Implied Volatility"] > 0]

    chunk["Mid Price"] = (chunk["Bid"] + chunk["Ask"]) / 2

    chunk = chunk.drop(
        columns=[c for c in DROP_COLS if c in chunk.columns],
        errors="ignore"
    )

    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        index=False,
        header=first_chunk
    )

    first_chunk = False

print("Filtering complete.")
print(f"Filtered dataset saved to {OUTPUT_FILE}")