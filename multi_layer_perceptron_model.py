import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

#SHAP
import os
os.makedirs("models", exist_ok=True)


print("=== MLP MODEL TRAINING (5-FOLD CV) ===")


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

    # Fit scaler only on training fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=42
    )

    mlp.fit(X_train_scaled, y_train)

    preds = mlp.predict(X_val_scaled)

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

final_scaler = StandardScaler()
X_scaled_full = final_scaler.fit_transform(X)

final_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

final_mlp.fit(X_scaled_full, y)

print("Final model trained on all data.")


# Save Predictions to a Full Dataset


full_predictions = pd.DataFrame({
    "Prediction": final_mlp.predict(X_scaled_full)
})

full_predictions.to_csv(
    "data/full_predictions_mlp.csv",
    index=False
)

# SHAP
import joblib
joblib.dump(final_mlp, "models/mlp_model.pkl")
joblib.dump(final_scaler, "models/mlp_scaler.pkl")


print("Saved full dataset predictions.")
print("=== MLP COMPLETE ===")