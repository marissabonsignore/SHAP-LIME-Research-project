import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

print("=== IMPROVED TCN MODEL TRAINING (5-FOLD CV) ===")

SEQ_LENGTH = 5

# ===============================
# Load Data
# ===============================

df = pd.read_csv("data/filtered_spx_options_with_features.csv")

df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values(["RIC", "Trade Date"]).reset_index(drop=True)

print(f"Loaded observations: {len(df)}")

# ===============================
# Build Sequences
# ===============================

X_sequences = []
y_targets = []

for ric, group in df.groupby("RIC"):
    iv_series = group["IV"].values
    price_series = group["Mid Price"].values

    if len(group) <= SEQ_LENGTH:
        continue

    for i in range(SEQ_LENGTH, len(group)):
        X_sequences.append(iv_series[i - SEQ_LENGTH:i])
        y_targets.append(price_series[i])

X = np.array(X_sequences)
y = np.array(y_targets)

print(f"Constructed sequences: {X.shape}")

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

    # -------------------------
    # Scale X inside fold
    # -------------------------
    x_scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, 1)
    X_val_2d = X_val.reshape(-1, 1)

    X_train_scaled = x_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val_2d).reshape(X_val.shape)

    X_train_scaled = X_train_scaled[..., np.newaxis]
    X_val_scaled = X_val_scaled[..., np.newaxis]

    # -------------------------
    # Scale y inside fold
    # -------------------------
    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1,1)).flatten()

    # -------------------------
    # Improved TCN Architecture
    # -------------------------
    model = Sequential([
        Input(shape=(SEQ_LENGTH, 1)),

        Conv1D(64, kernel_size=2, dilation_rate=1, padding="causal", activation="relu"),
        BatchNormalization(),

        Conv1D(64, kernel_size=2, dilation_rate=2, padding="causal", activation="relu"),
        BatchNormalization(),

        Conv1D(64, kernel_size=2, dilation_rate=4, padding="causal", activation="relu"),
        BatchNormalization(),

        GlobalAveragePooling1D(),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=50,
        batch_size=128,
        callbacks=[early_stop],
        verbose=0
    )

    # -------------------------
    # Predict & invert scaling
    # -------------------------
    preds_scaled = model.predict(X_val_scaled).flatten()
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

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

print("\nTraining final TCN on full dataset...")

x_scaler_full = StandardScaler()
X_full_2d = X.reshape(-1, 1)
X_scaled_full = x_scaler_full.fit_transform(X_full_2d).reshape(X.shape)
X_scaled_full = X_scaled_full[..., np.newaxis]

y_scaler_full = StandardScaler()
y_scaled_full = y_scaler_full.fit_transform(y.reshape(-1,1)).flatten()

final_model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),

    Conv1D(64, kernel_size=2, dilation_rate=1, padding="causal", activation="relu"),
    BatchNormalization(),

    Conv1D(64, kernel_size=2, dilation_rate=2, padding="causal", activation="relu"),
    BatchNormalization(),

    Conv1D(64, kernel_size=2, dilation_rate=4, padding="causal", activation="relu"),
    BatchNormalization(),

    GlobalAveragePooling1D(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(1)
])

final_model.compile(
    optimizer="adam",
    loss="mse"
)

final_model.fit(
    X_scaled_full,
    y_scaled_full,
    epochs=50,
    batch_size=128,
    verbose=0
)

print("Final TCN trained on all data.")
print("=== TCN COMPLETE ===")