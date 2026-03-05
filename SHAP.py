# shap_runner.py

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# =====================================================
# SETUP
# =====================================================

os.makedirs("shap_outputs", exist_ok=True)

FEATURES = ["Log_Moneyness", "Time_to_Maturity", "IV"]
TARGET = "Mid Price"

SEQ_LENGTH = 5
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

print("=== SHAP INTERPRETABILITY RUNNER ===")

# =====================================================
# LOAD DATA
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

# =====================================================
# TABULAR MODELS
# =====================================================
# ------------------ Linear Regression ------------------
print("\nLoading Linear Regression...")
lin_reg = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/tabular_scaler.pkl")

X_sample = X_test.iloc[:300]
X_sample_scaled = scaler.transform(X_sample)

# Background data should also be scaled
X_train_scaled = scaler.transform(X_train)

explainer_lin = shap.LinearExplainer(lin_reg, X_train_scaled)
shap_vals_lin = explainer_lin.shap_values(X_sample_scaled)

shap.summary_plot(shap_vals_lin, X_sample, show=False)
plt.savefig("shap_outputs/linear_summary.png", dpi=200)
plt.close()

print("Linear Regression SHAP done.")

lin_importance = np.abs(shap_vals_lin).mean(axis=0)
print("Linear Importance:", dict(zip(FEATURES, lin_importance)))

# ------------------ Gradient Boosting ------------------
print("\nLoading Gradient Boosting...")
gbr = joblib.load("models/gbr_model.pkl")

X_sample = X_test.iloc[:300]

explainer_gbr = shap.TreeExplainer(gbr)
shap_vals_gbr = explainer_gbr.shap_values(X_sample)

shap.summary_plot(shap_vals_gbr, X_sample, show=False)
plt.savefig("shap_outputs/gbr_summary.png", dpi=200)
plt.close()

print("GBR SHAP done.")

# Print numeric importance
gbr_importance = np.abs(shap_vals_gbr).mean(axis=0)
print("GBR Importance:", dict(zip(FEATURES, gbr_importance)))


# ------------------ Random Forest ------------------
print("\nLoading Random Forest...")
rf = joblib.load("models/rf_model.pkl")

X_sample = X_test.iloc[:300]

explainer_rf = shap.TreeExplainer(rf)
shap_vals_rf = explainer_rf.shap_values(X_sample)

shap.summary_plot(shap_vals_rf, X_sample, show=False)
plt.savefig("shap_outputs/rf_summary.png", dpi=200)
plt.close()

print("RF SHAP done.")

rf_importance = np.abs(shap_vals_rf).mean(axis=0)
print("RF Importance:", dict(zip(FEATURES, rf_importance)))


# ------------------ MLP ------------------
print("\nLoading MLP...")
mlp = joblib.load("models/mlp_model.pkl")

# Recreate scaler from TRAINING DATA (correct way)
scaler = StandardScaler()
scaler.fit(X_train)

X_sample = X_test.iloc[:100]
X_sample_scaled = scaler.transform(X_sample)

explainer_mlp = shap.KernelExplainer(
    mlp.predict,
    X_sample_scaled[:50]
)

shap_vals_mlp = explainer_mlp.shap_values(
    X_sample_scaled,
    silent=False
)

shap.summary_plot(shap_vals_mlp, X_sample, show=False)
plt.savefig("shap_outputs/mlp_summary.png", dpi=200)
plt.close()

print("MLP SHAP done.")

mlp_importance = np.abs(shap_vals_mlp).mean(axis=0)
print("MLP Importance:", dict(zip(FEATURES, mlp_importance)))

# =====================================================
# SEQUENCE MODELS (ROBUST VERSION)
# =====================================================

print("\nBuilding sequences for LSTM/TCN...")

df2 = df.sort_values(["RIC", "Trade Date"]).reset_index(drop=True)

# ------------------ Load LSTM First To Detect Shape ------------------
print("\nLoading LSTM...")
lstm = load_model("models/lstm_model.h5", compile=False)

print("LSTM input shape:", lstm.input_shape)

SEQ_LENGTH = lstm.input_shape[1]
N_FEATURES = lstm.input_shape[2]

print("Detected sequence length:", SEQ_LENGTH)
print("Detected feature count:", N_FEATURES)

# ---------------------------------------------------
# Build sequences matching EXACT model input shape
# ---------------------------------------------------

if N_FEATURES == 1:
    feature_cols = ["IV"]
else:
    feature_cols = FEATURES[:N_FEATURES]

X_seq = []

for ric, group in df2.groupby("RIC"):

    feature_matrix = group[feature_cols].values

    if len(group) <= SEQ_LENGTH:
        continue

    for i in range(SEQ_LENGTH, len(group)):
        X_seq.append(feature_matrix[i-SEQ_LENGTH:i])

X_seq = np.array(X_seq)

# Scale correctly
scaler_seq = StandardScaler()
X_seq_scaled = scaler_seq.fit_transform(
    X_seq.reshape(-1, N_FEATURES)
).reshape(X_seq.shape)

# Train/test split
n = len(X_seq_scaled)
train_end = int((1 - TEST_SPLIT - VAL_SPLIT) * n)
val_end = int((1 - TEST_SPLIT) * n)

X_test_seq = X_seq_scaled[val_end:]

# Flatten for SHAP
X_test_seq_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

# ------------------ LSTM SHAP ------------------

def lstm_predict(x):
    x_reshaped = x.reshape(-1, SEQ_LENGTH, N_FEATURES)
    return lstm.predict(x_reshaped, verbose=0).flatten()

X_sample_flat = X_test_seq_flat[:50]

explainer_lstm = shap.KernelExplainer(
    lstm_predict,
    X_sample_flat[:25]
)

shap_vals_lstm = explainer_lstm.shap_values(
    X_sample_flat,
    silent=False
)

shap.summary_plot(
    shap_vals_lstm,
    X_sample_flat,
    show=False
)

plt.savefig("shap_outputs/lstm_summary.png", dpi=200)
plt.close()

print("LSTM SHAP done.")


# ------------------ TCN ------------------

print("\nLoading TCN...")
tcn = load_model("models/tcn_model.h5", compile=False)

print("TCN input shape:", tcn.input_shape)

TCN_SEQ_LENGTH = tcn.input_shape[1]
TCN_FEATURES = tcn.input_shape[2]

print("TCN sequence length:", TCN_SEQ_LENGTH)
print("TCN feature count:", TCN_FEATURES)

# Build TCN sequences separately
if TCN_FEATURES == 1:
    tcn_feature_cols = ["IV"]
else:
    tcn_feature_cols = FEATURES[:TCN_FEATURES]

X_seq_tcn = []

for ric, group in df2.groupby("RIC"):

    feature_matrix = group[tcn_feature_cols].values

    if len(group) <= TCN_SEQ_LENGTH:
        continue

    for i in range(TCN_SEQ_LENGTH, len(group)):
        X_seq_tcn.append(feature_matrix[i-TCN_SEQ_LENGTH:i])

X_seq_tcn = np.array(X_seq_tcn)

# Scale correctly
scaler_seq_tcn = StandardScaler()
X_seq_tcn_scaled = scaler_seq_tcn.fit_transform(
    X_seq_tcn.reshape(-1, TCN_FEATURES)
).reshape(X_seq_tcn.shape)

n_tcn = len(X_seq_tcn_scaled)
train_end_tcn = int((1 - TEST_SPLIT - VAL_SPLIT) * n_tcn)
val_end_tcn = int((1 - TEST_SPLIT) * n_tcn)

X_test_seq_tcn = X_seq_tcn_scaled[val_end_tcn:]

# Flatten for SHAP
X_test_seq_tcn_flat = X_test_seq_tcn.reshape(X_test_seq_tcn.shape[0], -1)

def tcn_predict(x):
    x_reshaped = x.reshape(-1, TCN_SEQ_LENGTH, TCN_FEATURES)
    return tcn.predict(x_reshaped, verbose=0).flatten()

X_sample_flat_tcn = X_test_seq_tcn_flat[:50]

explainer_tcn = shap.KernelExplainer(
    tcn_predict,
    X_sample_flat_tcn[:25]
)

shap_vals_tcn = explainer_tcn.shap_values(
    X_sample_flat_tcn,
    silent=False
)

shap.summary_plot(
    shap_vals_tcn,
    X_sample_flat_tcn,
    show=False
)

plt.savefig("shap_outputs/tcn_summary.png", dpi=200)
plt.close()

print("TCN SHAP done.")