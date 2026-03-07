# lime_runner.py

import os
import numpy as np
import pandas as pd
import joblib

from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

print("=== STARTING LIME SCRIPT ===")

# =====================================================
# SETUP
# =====================================================

os.makedirs("lime_outputs", exist_ok=True)

FEATURES = [
    "Log_Moneyness",
    "IV"
]
TARGET = "Mid Price"

SEQ_LENGTH = 5
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

# =====================================================
# LOAD DATA (MATCHES SHAP)
# =====================================================

df = pd.read_csv("data/filtered_spx_options_with_features.csv")
df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values("Trade Date").reset_index(drop=True)

X = df[FEATURES]
y = df[TARGET]

n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train = X.iloc[:train_end]
X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

# =====================================================
# TABULAR MODELS
# =====================================================

print("\nLoading tabular models...")
gbr = joblib.load("models/gbr_model.pkl")
rf = joblib.load("models/rf_model.pkl")
mlp = joblib.load("models/mlp_model.pkl")
lin_reg = joblib.load("models/linear_regression_model.pkl")
scaler_tab = joblib.load("models/tabular_scaler.pkl")

print("Creating LIME explainer for tabular models...")

# Scale training data (must match model training)
X_train_scaled = scaler_tab.transform(X_train)
X_test_scaled = scaler_tab.transform(X_test)

lime_tab = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=FEATURES,
    mode="regression",
    discretize_continuous=True
)

EXPLAIN_IDXS = [0, 10, 50]

for i in EXPLAIN_IDXS:
    print(f"Explaining tabular index {i}...")

    x_raw = X_test.iloc[i].values
    x_scaled = scaler_tab.transform([x_raw])[0]

    # Linear Regression
    exp = lime_tab.explain_instance(x_scaled, lin_reg.predict, num_features=len(FEATURES))
    exp.save_to_file(f"lime_outputs/LINEAR_idx_{i}.html")

    # Gradient Boosting
    exp = lime_tab.explain_instance(x_scaled, gbr.predict, num_features=len(FEATURES))
    exp.save_to_file(f"lime_outputs/GBR_idx_{i}.html")

    # Random Forest
    exp = lime_tab.explain_instance(x_scaled, rf.predict, num_features=len(FEATURES))
    exp.save_to_file(f"lime_outputs/RF_idx_{i}.html")

    # MLP
    exp = lime_tab.explain_instance(x_scaled, mlp.predict, num_features=len(FEATURES))
    exp.save_to_file(f"lime_outputs/MLP_idx_{i}.html")

print("Tabular LIME complete.")

# =====================================================
# SEQUENCE MODELS
# =====================================================

print("\nBuilding sequences for LSTM/TCN...")

df2 = df.sort_values(["RIC", "Trade Date"]).reset_index(drop=True)

# ------------------ LSTM ------------------

print("\nLoading LSTM...")
lstm = load_model("models/lstm_model.h5", compile=False)

LSTM_SEQ_LENGTH = lstm.input_shape[1]
LSTM_FEATURES = lstm.input_shape[2]

if LSTM_FEATURES == 1:
    lstm_feature_cols = ["IV"]
else:
    lstm_feature_cols = FEATURES[:LSTM_FEATURES]

X_seq_lstm = []
y_seq = []

for ric, group in df2.groupby("RIC"):
    feature_matrix = group[lstm_feature_cols].values
    price = group["Mid Price"].values

    if len(group) <= LSTM_SEQ_LENGTH:
        continue

    for t in range(LSTM_SEQ_LENGTH, len(group)):
        X_seq_lstm.append(feature_matrix[t-LSTM_SEQ_LENGTH:t])
        y_seq.append(price[t])

X_seq_lstm = np.array(X_seq_lstm)
y_seq = np.array(y_seq)

scaler_lstm = StandardScaler()
X_seq_lstm_scaled = scaler_lstm.fit_transform(
    X_seq_lstm.reshape(-1, LSTM_FEATURES)
).reshape(X_seq_lstm.shape)

N = len(X_seq_lstm_scaled)
train_end = int((1 - TEST_SPLIT - VAL_SPLIT) * N)
val_end = int((1 - TEST_SPLIT) * N)

X_train_lstm = X_seq_lstm_scaled[:train_end]
X_test_lstm = X_seq_lstm_scaled[val_end:]
y_test_seq = y_seq[val_end:]

def lstm_predict(x):
    x3d = np.array(x).reshape(-1, LSTM_SEQ_LENGTH, LSTM_FEATURES)
    return lstm.predict(x3d, verbose=0).flatten()

# Correct flattened feature names
lstm_feature_names = []
for lag in range(LSTM_SEQ_LENGTH, 0, -1):
    for feat in lstm_feature_cols:
        lstm_feature_names.append(f"{feat}_lag_{lag}")

lime_lstm = LimeTabularExplainer(
    training_data=X_train_lstm.reshape(len(X_train_lstm), -1),
    feature_names=lstm_feature_names,
    mode="regression",
    discretize_continuous=True
)

for i in EXPLAIN_IDXS:
    print(f"Explaining LSTM index {i}...")
    x = X_test_lstm[i].reshape(-1)
    exp = lime_lstm.explain_instance(x, lstm_predict, num_features=len(lstm_feature_names))
    exp.save_to_file(f"lime_outputs/LSTM_idx_{i}.html")

print("LSTM LIME complete.")

# ------------------ TCN ------------------

print("\nLoading TCN...")
tcn = load_model("models/tcn_model.h5", compile=False)

TCN_SEQ_LENGTH = tcn.input_shape[1]
TCN_FEATURES = tcn.input_shape[2]

if TCN_FEATURES == 1:
    tcn_feature_cols = ["IV"]
else:
    tcn_feature_cols = FEATURES[:TCN_FEATURES]

X_seq_tcn = []

for ric, group in df2.groupby("RIC"):
    feature_matrix = group[tcn_feature_cols].values

    if len(group) <= TCN_SEQ_LENGTH:
        continue

    for t in range(TCN_SEQ_LENGTH, len(group)):
        X_seq_tcn.append(feature_matrix[t-TCN_SEQ_LENGTH:t])

X_seq_tcn = np.array(X_seq_tcn)

scaler_tcn = StandardScaler()
X_seq_tcn_scaled = scaler_tcn.fit_transform(
    X_seq_tcn.reshape(-1, TCN_FEATURES)
).reshape(X_seq_tcn.shape)

X_train_tcn = X_seq_tcn_scaled[:train_end]
X_test_tcn = X_seq_tcn_scaled[val_end:]

def tcn_predict(x):
    x3d = np.array(x).reshape(-1, TCN_SEQ_LENGTH, TCN_FEATURES)
    return tcn.predict(x3d, verbose=0).flatten()

# Correct flattened feature names
tcn_feature_names = []
for lag in range(TCN_SEQ_LENGTH, 0, -1):
    for feat in tcn_feature_cols:
        tcn_feature_names.append(f"{feat}_lag_{lag}")

lime_tcn = LimeTabularExplainer(
    training_data=X_train_tcn.reshape(len(X_train_tcn), -1),
    feature_names=tcn_feature_names,
    mode="regression",
    discretize_continuous=True
)

for i in EXPLAIN_IDXS:
    print(f"Explaining TCN index {i}...")
    x = X_test_tcn[i].reshape(-1)
    exp = lime_tcn.explain_instance(x, tcn_predict, num_features=len(tcn_feature_names))
    exp.save_to_file(f"lime_outputs/TCN_idx_{i}.html")

print("TCN LIME complete.")

print("\n=== LIME COMPLETE ===")