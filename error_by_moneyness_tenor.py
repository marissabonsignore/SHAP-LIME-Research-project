import pandas as pd
import numpy as np

print("=== ERROR ANALYSIS BY MONEyness AND TENOR (ALL MODELS) ===")

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------
DATA_FILE = "data/filtered_spx_options_with_features.csv"

# 🔴 IMPORTANT:
# Add new models here when you create them
PREDICTION_FILES = {
    "Gradient Boosting": "data/test_predictions_gb.csv",
    "Random Forest": "data/test_predictions_rf.csv",
    "MLP": "data/test_predictions_mlp.csv"
}

# --------------------------------------------------
# LOAD FULL DATASET
# --------------------------------------------------
df = pd.read_csv(DATA_FILE)

n = len(df)
test_start = int(0.85 * n)
df_test = df.iloc[test_start:].reset_index(drop=True)

print(f"Total test observations: {len(df_test)}")

# --------------------------------------------------
# BUCKET DEFINITIONS
# --------------------------------------------------
def moneyness_bucket(x):
    if x < -0.2:
        return "Deep OTM"
    elif x < -0.05:
        return "OTM"
    elif x <= 0.05:
        return "ATM"
    elif x <= 0.2:
        return "ITM"
    else:
        return "Deep ITM"

def tenor_bucket(T):
    if T <= 0.25:
        return "Short"
    elif T <= 1.0:
        return "Medium"
    else:
        return "Long"

df_test["Moneyness_Bucket"] = df_test["Log_Moneyness"].apply(moneyness_bucket)
df_test["Tenor_Bucket"] = df_test["Time_to_Maturity"].apply(tenor_bucket)

# --------------------------------------------------
# LOOP OVER MODELS
# --------------------------------------------------
for model_name, pred_file in PREDICTION_FILES.items():

    print(f"\n--- {model_name} ---")

    preds = pd.read_csv(pred_file).reset_index(drop=True)

    if len(preds) != len(df_test):
        raise ValueError(f"Prediction length mismatch for {model_name}")

    df_model = df_test.copy()
    df_model["Prediction"] = preds["Prediction"]

    # Errors
    df_model["Abs_Error"] = np.abs(df_model["Mid Price"] - df_model["Prediction"])
    df_model["Sq_Error"] = (df_model["Mid Price"] - df_model["Prediction"]) ** 2

    # --------------------------------------------------
    # ERROR BY MONEyness
    # --------------------------------------------------
    moneyness_summary = (
        df_model
        .groupby("Moneyness_Bucket")
        .agg(
            MAE=("Abs_Error", "mean"),
            RMSE=("Sq_Error", lambda x: np.sqrt(x.mean())),
            Count=("Abs_Error", "count")
        )
    )

    # --------------------------------------------------
    # ERROR BY TENOR
    # --------------------------------------------------
    tenor_summary = (
        df_model
        .groupby("Tenor_Bucket")
        .agg(
            MAE=("Abs_Error", "mean"),
            RMSE=("Sq_Error", lambda x: np.sqrt(x.mean())),
            Count=("Abs_Error", "count")
        )
    )

    print("\nError by Moneyness:")
    print(moneyness_summary)

    print("\nError by Tenor:")
    print(tenor_summary)

print("\n=== ERROR ANALYSIS COMPLETE ===")