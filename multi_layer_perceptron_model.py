import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("=== MLP MODEL TRAINING ===")

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

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=200,
    random_state=42
)

print("Training MLP model...")
mlp.fit(X_train_scaled, y_train)

def evaluate(model, X, y, label):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    print(f"{label} RMSE: {rmse:.4f}")
    print(f"{label} MAE:  {mae:.4f}")

print("\nModel performance:")
evaluate(mlp, X_train_scaled, y_train, "Train")
evaluate(mlp, X_val_scaled, y_val, "Validation")
evaluate(mlp, X_test_scaled, y_test, "Test")

print("=== MLP COMPLETE ===")

test_predictions = pd.DataFrame({
    "Prediction": mlp.predict(X_test_scaled)
})

test_predictions.to_csv(
    "data/test_predictions_mlp.csv",
    index=False
)

print("Saved MLP test predictions.")
