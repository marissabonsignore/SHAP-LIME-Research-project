import pandas as pd
import glob
import os

data_path = "data/kpydishe_mbosign_sbaduge_aaqeel_aqaisar_tlonon/*.csv"

csv_files = glob.glob(data_path)

print(f"Found {len(csv_files)} CSV files.")

df_list = []

for file in csv_files:
    df = pd.read_csv(file)
    df["source_file"] = os.path.basename(file) 
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

merged_df["Trade Date"] = pd.to_datetime(merged_df["Trade Date"])
merged_df = merged_df.sort_values("Trade Date")

merged_df.to_csv("merged_spx_options.csv", index=False)

print("Merged file saved as merged_spx_options.csv")