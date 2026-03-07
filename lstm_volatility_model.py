import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

# SHAP
import os
os.makedirs("models", exist_ok=True)


print("=== LSTM MODEL TRAINING (ALIGNED, 5-FOLD CV) ===")

SEQ_LENGTH = 5

# ===============================
# Load Data
# ===============================

df = pd.read_csv("data/filtered_spx_options_with_features.csv")

df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values(["RIC", "Trade Date"]).reset_index(drop=True)

print(f"Loaded observations: {len(df)}")

FEATURES = [
    "Log_Moneyness",
    "IV"
]

TARGET = "Mid Price"

# ===============================
# Build Sequences (ALL FEATURES)
# ===============================

X_sequences = []
y_targets = []

for ric, group in df.groupby("RIC"):

    feature_matrix = group[FEATURES].values
    price_series = group[TARGET].values

    if len(group) <= SEQ_LENGTH:
        continue

    for i in range(SEQ_LENGTH, len(group)):
        X_sequences.append(feature_matrix[i-SEQ_LENGTH:i])
        y_targets.append(price_series[i])

X = np.array(X_sequences)
y = np.array(y_targets)

print(f"Constructed sequences: {X.shape}")
# Expected shape: (samples, 5, 7)

# ===============================
# 5-Fold Cross Validation
# ===============================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []

print("\nStarting 5-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

    print(f"\n--- Fold {fold+1} ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Scale features inside fold
    scaler = StandardScaler()

    # reshape to 2D for scaling
    X_train_2d = X_train.reshape(-1, X_train.shape[2])
    X_val_2d = X_val.reshape(-1, X_val.shape[2])

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)

    # Build model
    model = Sequential([
        Input(shape=(SEQ_LENGTH, len(FEATURES))),
        LSTM(32),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=30,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )

    preds = model.predict(X_val_scaled).flatten()

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)

    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    print(f"Fold {fold+1} MAE:  {mae:.4f}")

    rmse_scores.append(rmse)
    mae_scores.append(mae)

# ===============================
# CV Summary
# ===============================

print("\n=== 5-Fold CV Results ===")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"RMSE Std Dev: {np.std(rmse_scores):.4f}")
print(f"Average MAE:  {np.mean(mae_scores):.4f}")
print(f"MAE Std Dev:  {np.std(mae_scores):.4f}")

# ===============================
# Train Final Model on Full Data
# ===============================

print("\nTraining final LSTM on full dataset...")

scaler_full = StandardScaler()

X_full_2d = X.reshape(-1, X.shape[2])
X_scaled_full = scaler_full.fit_transform(X_full_2d).reshape(X.shape)

final_model = Sequential([
    Input(shape=(SEQ_LENGTH, len(FEATURES))),
    LSTM(32),
    Dense(1)
])

final_model.compile(
    optimizer="adam",
    loss="mse"
)

final_model.fit(
    X_scaled_full,
    y,
    epochs=30,
    batch_size=256,
    verbose=0
)

# SHAP
import joblib
joblib.dump(scaler_full, "models/lstm_scaler.pkl")
final_model.save("models/lstm_model.h5")

print("Final LSTM trained on all data.")
print("=== LSTM COMPLETE ===")