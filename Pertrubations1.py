import numpy as np
import pandas as pd


# -----------------------------
# LOAD DATA
# -----------------------------
def load_spx_data(csv_path="filtered_spx_options_with_features.csv"):
    FEATURES = ["Log_Moneyness", "Time_to_Maturity", "IV"]

    df = pd.read_csv(
        csv_path,
        usecols=FEATURES,
        engine="c"
    )

    return df


# -----------------------------
# GAUSSIAN PERTURBATION
# -----------------------------
def generate_gaussian_perturbations(x_instance, n_samples=100, noise_scale=0.1):
    x_instance = np.asarray(x_instance).flatten()
    n_features = len(x_instance)

    noise = np.random.normal(
        loc=0.0,
        scale=noise_scale,
        size=(n_samples, n_features)
    )

    perturbed_samples = x_instance.reshape(1, -1) + noise
    return perturbed_samples


# -----------------------------
# IV SHOCK
# -----------------------------
def generate_iv_shock(x_instance, shock_percent):
    x_instance = np.asarray(x_instance).flatten()
    shocked = x_instance.copy()

    # IV is column index 2
    shocked[2] = shocked[2] * (1 + shock_percent)

    return shocked.reshape(1, -1)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # Load dataset
    df = load_spx_data("filtered_spx_options_with_features.csv")

    print("\nORIGINAL DATAFRAME (first 5 rows):")
    print(df.head())

    # Select first row
    base_row = df.iloc[0].values

    print("\nBASE INSTANCE:")
    print(df.iloc[[0]])

    # -----------------------------
    # Gaussian Perturbations
    # -----------------------------
    gaussian_samples = generate_gaussian_perturbations(
        x_instance=base_row,
        n_samples=5,   # just 5 for clean display
        noise_scale=0.05
    )

    df_gaussian = pd.DataFrame(gaussian_samples, columns=df.columns)

    print("\nGAUSSIAN PERTURBATIONS:")
    print(df_gaussian)

    # -----------------------------
    # IV +20% Shock
    # -----------------------------
    iv_up = generate_iv_shock(base_row, shock_percent=0.20)
    df_iv_up = pd.DataFrame(iv_up, columns=df.columns)

    print("\nIV +20% SHOCK:")
    print(df_iv_up)

    # -----------------------------
    # IV -20% Shock
    # -----------------------------
    iv_down = generate_iv_shock(base_row, shock_percent=-0.20)
    df_iv_down = pd.DataFrame(iv_down, columns=df.columns)

    print("\nIV -20% SHOCK:")
    print(df_iv_down)
