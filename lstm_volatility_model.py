import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("=== LSTM VOLATILITY MODEL TRAINING ===")

SEQ_LENGTH = 5   
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

df = pd.read_csv("data/filtered_spx_options_with_features.csv")

df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values(["RIC", "Trade Date"]).reset_index(drop=True)

print(f"Loaded observations: {len(df)}")

X_sequences = []
y_targets = []

for ric, group in df.groupby("RIC"):
    iv_series = group["IV"].values
    price_series = group["Mid Price"].values

    if len(group) <= SEQ_LENGTH:
        continue

    for i in range(SEQ_LENGTH, len(group)):
        X_sequences.append(iv_series[i-SEQ_LENGTH:i])
        y_targets.append(price_series[i])

X = np.array(X_sequences)
y = np.array(y_targets)

print(f"Constructed sequences: {X.shape}")

scaler = StandardScaler()
X_reshaped = X.reshape(-1, 1)
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

n = len(X)
train_end = int((1 - TEST_SPLIT - VAL_SPLIT) * n)
val_end = int((1 - TEST_SPLIT) * n)

X_train, y_train = X_scaled[:train_end], y[:train_end]
X_val, y_val = X_scaled[train_end:val_end], y[train_end:val_end]
X_test, y_test = X_scaled[val_end:], y[val_end:]

print("Split sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

model = Sequential([
    LSTM(32, input_shape=(SEQ_LENGTH, 1)),
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

print("Training LSTM...")
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

preds = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("\nLSTM Model Performance:")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")

print("=== LSTM COMPLETE ===")