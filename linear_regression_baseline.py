from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import joblib

print("=== LINEAR REGRESSION MODEL TRAINING (ALIGNED, 5-FOLD CV) ===")

# Load data
df = pd.read_csv("data/filtered_spx_options_with_features.csv")
df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values("Trade Date").reset_index(drop=True)

FEATURES = [
    "Log_Moneyness",
    "IV"
]
TARGET = "Mid Price"

X = df[FEATURES].values
y = df[TARGET].values

print(f"Loaded observations: {len(X)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []

print("\nStarting 5-Fold Cross Validation...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):

    print(f"\n--- Fold {fold} ---")

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)

    print(f"Fold {fold} RMSE: {rmse:.4f}")
    print(f"Fold {fold} MAE: {mae:.4f}")

print("\n=== 5-Fold CV Results ===")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"RMSE Std Dev: {np.std(rmse_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f}")
print(f"MAE Std Dev: {np.std(mae_scores):.4f}")

# Save linear regression model

print("\nTraining final Linear Regression on full dataset...")

# Scale full dataset
from sklearn.preprocessing import StandardScaler

scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

# Train final model
final_lin = LinearRegression()
final_lin.fit(X_scaled_full, y)

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(final_lin, "models/linear_regression_model.pkl")

# Save scaler (VERY important)
joblib.dump(scaler_full, "models/tabular_scaler.pkl")

print("Saved linear_regression_model.pkl")
print("Saved tabular_scaler.pkl")