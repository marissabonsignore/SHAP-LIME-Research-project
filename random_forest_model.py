import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=== RANDOM FOREST MODEL TRAINING ===")

df = pd.read_csv("data/filtered_spx_options_with_features.csv")

df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values("Trade Date").reset_index(drop=True)

print(f"Loaded observations: {len(df)}")

FEATURES = [
    "Log_Moneyness",
    "Time_to_Maturity",
    "IV"
]

TARGET = "Mid Price"

X = df[FEATURES]
y = df[TARGET]

n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("Split sizes:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42
)

print("Training Random Forest model...")
rf.fit(X_train, y_train)

def evaluate(model, X, y, label):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    print(f"{label} RMSE: {rmse:.4f}")
    print(f"{label} MAE:  {mae:.4f}")

print("\nModel performance:")
evaluate(rf, X_train, y_train, "Train")
evaluate(rf, X_val, y_val, "Validation")
evaluate(rf, X_test, y_test, "Test")

print("=== RANDOM FOREST COMPLETE ===")

test_predictions = pd.DataFrame({
    "Prediction": rf.predict(X_test)
})

test_predictions.to_csv(
    "data/test_predictions_rf.csv",
    index=False
)

print("Saved Random Forest test predictions.")