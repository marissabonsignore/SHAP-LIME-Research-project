# shap_runner.py

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Create output folder
os.makedirs("shap_outputs", exist_ok=True)

FEATURES = ["Log_Moneyness", "Time_to_Maturity", "IV"]
TARGET = "Mid Price"

SEQ_LENGTH = 5
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

print("=== SHAP INTERPRETABILITY RUNNER ===")

# =====================================================
# TABULAR MODELS (GBR, RF, MLP)
# =====================================================

df = pd.read_csv("data/filtered_spx_options_with_features.csv")
df["Trade Date"] = pd.to_datetime(df["Trade Date"])
df = df.sort_values("Trade Date").reset_index(drop=True)

X = df[FEATURES]
y = df[TARGET]

n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_test = X.iloc[val_end:]

# ------------------ Gradient Boosting ------------------
print("\nLoading Gradient Boosting...")
gbr = joblib.load("models/gbr_model.pkl")

explainer_gbr = shap.TreeExplainer(gbr)
shap_vals_gbr = explainer_gbr.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_vals_gbr, X_test, show=False)
plt.savefig("shap_outputs/gbr_summary.png", dpi=200)
plt.close()

# ------------------ Random Forest ------------------
print("Loading Random Forest...")
rf = joblib.load("models/rf_model.pkl")

explainer_rf = shap.TreeExplainer(rf)
shap_vals_rf = explainer_rf.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_vals_rf, X_test, show=False)
plt.savefig("shap_outputs/rf_summary.png", dpi=200)
plt.close()

# ------------------ MLP ------------------
print("Loading MLP...")
mlp = joblib.load("models/mlp_model.pkl")
scaler = joblib.load("models/mlp_scaler.pkl")

X_test_scaled = scaler.transform(X_test)

explainer_mlp = shap.KernelExplainer(
    mlp.predict,
    X_test_scaled[:200]
)

shap_vals_mlp = explainer_mlp.shap_values(X_test_scaled[:300])

plt.figure()
shap.summary_plot(shap_vals_mlp, X_test.iloc[:300], show=False)
plt.savefig("shap_outputs/mlp_summary.png", dpi=200)
plt.close()

# =====================================================
# SEQUENCE MODELS (LSTM + TCN)
# =====================================================

print("\nBuilding sequences for LSTM/TCN...")

df2 = pd.read_csv("data/filtered_spx_options_with_features.csv")
df2["Trade Date"] = pd.to_datetime(df2["Trade Date"])
df2 = df2.sort_values(["RIC", "Trade Date"]).reset_index(drop=True)

X_seq = []
y_seq = []

for ric, group in df2.groupby("RIC"):
    iv = group["IV"].values
    price = group["Mid Price"].values

    if len(group) <= SEQ_LENGTH:
        continue

    for i in range(SEQ_LENGTH, len(group)):
        X_seq.append(iv[i-SEQ_LENGTH:i])
        y_seq.append(price[i])

X_seq = np.array(X_seq)

scaler_seq = StandardScaler()
X_seq_scaled = scaler_seq.fit_transform(X_seq.reshape(-1,1)).reshape(X_seq.shape)

n = len(X_seq_scaled)
train_end = int((1 - TEST_SPLIT - VAL_SPLIT) * n)
val_end = int((1 - TEST_SPLIT) * n)

X_test_seq = X_seq_scaled[val_end:]
X_test_seq_3d = X_test_seq.reshape(-1, SEQ_LENGTH, 1)

# ------------------ LSTM ------------------
print("Loading LSTM...")
lstm = load_model("models/lstm_model.h5")

def lstm_predict(x):
    x3d = np.array(x).reshape(-1, SEQ_LENGTH, 1)
    return lstm.predict(x3d).flatten()

explainer_lstm = shap.KernelExplainer(
    lstm_predict,
    X_test_seq[:200]
)

shap_vals_lstm = explainer_lstm.shap_values(X_test_seq[:300])

plt.figure()
shap.summary_plot(shap_vals_lstm, X_test_seq[:300], show=False)
plt.savefig("shap_outputs/lstm_summary.png", dpi=200)
plt.close()

# ------------------ TCN ------------------
print("Loading TCN...")
tcn = load_model("models/tcn_model.h5")

def tcn_predict(x):
    x3d = np.array(x).reshape(-1, SEQ_LENGTH, 1)
    return tcn.predict(x3d).flatten()

explainer_tcn = shap.KernelExplainer(
    tcn_predict,
    X_test_seq[:200]
)

shap_vals_tcn = explainer_tcn.shap_values(X_test_seq[:300])

plt.figure()
shap.summary_plot(shap_vals_tcn, X_test_seq[:300], show=False)
plt.savefig("shap_outputs/tcn_summary.png", dpi=200)
plt.close()

print("\nAll SHAP plots saved in shap_outputs/")
