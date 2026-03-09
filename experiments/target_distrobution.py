import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"DATA/admin1_dataset.parquet"
df = pd.read_parquet(DATA_PATH)

df["year_month"] = pd.to_datetime(df["year_month"])

TARGET = "ipc_phase_fews"

# same temporal split
TEST_MONTHS = 6
unique_months = np.sort(df["year_month"].unique())
test_months = unique_months[-TEST_MONTHS:]

train_mask = ~df["year_month"].isin(test_months)
test_mask = df["year_month"].isin(test_months)

# --------------------
# Plot 1 – Overall distribution
# --------------------
overall_counts = df[TARGET].value_counts().sort_index()

plt.figure()
overall_counts.plot(kind="bar")
plt.title("Overall Target Distribution (IPC Phase)")
plt.xlabel("IPC Phase")
plt.ylabel("Count")
plt.show()

# --------------------
# Plot 2 – Train distribution
# --------------------
train_counts = df.loc[train_mask, TARGET].value_counts().sort_index()

plt.figure()
train_counts.plot(kind="bar")
plt.title("Train Target Distribution (IPC Phase)")
plt.xlabel("IPC Phase")
plt.ylabel("Count")
plt.show()

# --------------------
# Plot 3 – Test distribution
# --------------------
test_counts = df.loc[test_mask, TARGET].value_counts().sort_index()

plt.figure()
test_counts.plot(kind="bar")
plt.title("Test Target Distribution (IPC Phase)")
plt.xlabel("IPC Phase")
plt.ylabel("Count")
plt.show()
