import pandas as pd

df = pd.read_csv("data/filtered_spx_options_with_features.csv")

df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values("Trade Date").reset_index(drop=True)

print("Total observations:", len(df))
print("\nColumns in dataset:")
print(df.columns.tolist())

FEATURES = [
    "Log_Moneyness",
    "Time_to_Maturity",
    "IV"
]

TARGET = "Mid Price"

X = df[FEATURES]
y = df[TARGET]

print("\nSelected features (X):", FEATURES)
print("Target (y):", TARGET)

print("\nFeature sample:")
print(X.head())

print("\nTarget sample:")
print(y.head())

n = len(df)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("\nSplit sizes:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

print("\nDate ranges:")
print("Train:", df.loc[:train_end - 1, "Trade Date"].min(), "→", df.loc[:train_end - 1, "Trade Date"].max())
print("Validation:", df.loc[train_end:val_end - 1, "Trade Date"].min(), "→", df.loc[train_end:val_end - 1, "Trade Date"].max())
print("Test:", df.loc[val_end:, "Trade Date"].min(), "→", df.loc[val_end:, "Trade Date"].max())