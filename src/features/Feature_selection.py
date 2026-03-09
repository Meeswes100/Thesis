import pandas as pd

# Load data
df = pd.read_parquet(r"DATA\part_0000.parquet")

# ----------------------------
# 1) Drop rows where FEWS target is missing
# ----------------------------
df = df[df["ipc_phase_fews"].notna()].copy()

print("New shape after dropping missing FEWS:", df.shape)

# ----------------------------
# 2) Calculate missing percentage per column
# ----------------------------
missing_share = df.isna().mean()

# ----------------------------
# 3) Select columns where > 50% missing
# ----------------------------
cols_over_50_missing = missing_share[missing_share > 0.0]

print("\nColumns with more than 50% missing values:")
print(cols_over_50_missing.sort_values(ascending=False))

print("\nNumber of columns with >50% missing:", len(cols_over_50_missing))
