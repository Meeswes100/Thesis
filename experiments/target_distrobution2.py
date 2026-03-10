import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = r"C:\Users\meesw\projects\Thesis\DATA\admin1_dataset_v2.parquet"
df = pd.read_parquet(DATA_PATH)

df["year_month"] = pd.to_datetime(df["year_month"])
TARGET = "ipc_phase_fews"

# -----------------------------
# Time-based split (last N months as test)
# -----------------------------
TEST_MONTHS = 12
unique_months = np.sort(df["year_month"].unique())
if len(unique_months) <= TEST_MONTHS:
    raise ValueError(f"Not enough months ({len(unique_months)}) for TEST_MONTHS={TEST_MONTHS}")

test_months = unique_months[-TEST_MONTHS:]
train_mask = ~df["year_month"].isin(test_months)
test_mask = df["year_month"].isin(test_months)

# -----------------------------
# Compute distributions
# -----------------------------
train_counts = df.loc[train_mask, TARGET].value_counts().sort_index()
test_counts  = df.loc[test_mask, TARGET].value_counts().sort_index()

# Ensure same index (phases) in both
all_phases = sorted(set(train_counts.index).union(set(test_counts.index)))
train_counts = train_counts.reindex(all_phases, fill_value=0)
test_counts  = test_counts.reindex(all_phases, fill_value=0)

dist_counts = pd.DataFrame({"Train": train_counts, "Test": test_counts})

# Percentages (often better for slides)
train_pct = train_counts / train_counts.sum() * 100
test_pct  = test_counts / test_counts.sum() * 100
dist_pct = pd.DataFrame({"Train %": train_pct, "Test %": test_pct})

# -----------------------------
# Plot: Train vs Test (Counts)
# -----------------------------
ax = dist_counts.plot(kind="bar")
plt.title(f"Train vs Test Target Distribution (Counts) | Test = last {TEST_MONTHS} months")
plt.xlabel("IPC Phase")
plt.ylabel("Count")
plt.legend(title="")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Train vs Test (Percent)
# -----------------------------
ax = dist_pct.plot(kind="bar")
plt.title(f"Train vs Test Target Distribution (%) | Test = last {TEST_MONTHS} months")
plt.xlabel("IPC Phase")
plt.ylabel("Percentage (%)")
plt.legend(title="")
plt.tight_layout()
plt.show()

# -----------------------------
# Optional: print summary table
# -----------------------------
summary = pd.concat(
    [
        dist_counts.rename(columns={"Train": "Train_count", "Test": "Test_count"}),
        dist_pct.rename(columns={"Train %": "Train_pct", "Test %": "Test_pct"}),
    ],
    axis=1,
)
print("\n--- Distribution summary ---")
print(summary)