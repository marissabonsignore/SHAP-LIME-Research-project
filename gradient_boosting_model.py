import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

# SHAP
import os
os.makedirs("models", exist_ok=True)

print("=== GRADIENT BOOSTING MODEL TRAINING (5-FOLD CV) ===")

# Load Data

df = pd.read_csv("data/filtered_spx_options_with_features.csv")

df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values("Trade Date").reset_index(drop=True)

print(f"Loaded observations: {len(df)}")

# Define Features and Target

FEATURES = [
    "Log_Moneyness",
    "IV"
]

TARGET = "Mid Price"

X = df[FEATURES].values
y = df[TARGET].values

# 5-Fold Cross Validation

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []

print("\nStarting 5-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

    print(f"\n--- Fold {fold+1} ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    gbr.fit(X_train, y_train)

    preds = gbr.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)

    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    print(f"Fold {fold+1} MAE:  {mae:.4f}")

    rmse_scores.append(rmse)
    mae_scores.append(mae)


# Results Summary

print("\n=== 5-Fold CV Results ===")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"RMSE Std Dev: {np.std(rmse_scores):.4f}")
print(f"Average MAE:  {np.mean(mae_scores):.4f}")
print(f"MAE Std Dev:  {np.std(mae_scores):.4f}")


# Train Final Model on All Data


print("\nTraining final model on full dataset...")

final_gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

final_gbr.fit(X, y)

print("Final model trained on all data.")


# Save Predictions to a Full Dataset

full_predictions = pd.DataFrame({
    "Prediction": final_gbr.predict(X)
})

full_predictions.to_csv(
    "data/full_predictions_gb.csv",
    index=False
)

#SHAP
import joblib
joblib.dump(gbr, "models/gbr_model.pkl")

print("Saved full dataset predictions.")
print("=== GRADIENT BOOSTING COMPLETE ===")