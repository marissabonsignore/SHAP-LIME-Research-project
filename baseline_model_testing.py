import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/filtered_spx_options_with_features.csv")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

y = df['mid_price'].dropna()

mean_pred = np.full(len(y), y.mean())

baseline_rmse = np.sqrt(mean_squared_error(y, mean_pred))
print("Baseline RMSE (mean predictor):", baseline_rmse)