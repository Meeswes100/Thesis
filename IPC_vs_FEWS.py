import pandas as pd
import numpy as np

# ----------------------------
# Load
# ----------------------------
df = pd.read_parquet(r"DATA\part_0000.parquet")

# Column names you confirmed
TIME  = "year_month"
A1    = "ADMIN1"
A2    = "ADMIN2"
FEWS  = "ipc_phase_fews"
IPCCH = "ipc_phase_ipcch"

# Make sure targets are numeric
df[FEWS]  = pd.to_numeric(df[FEWS], errors="coerce")
df[IPCCH] = pd.to_numeric(df[IPCCH], errors="coerce")

print("Dataset shape:", df.shape)

# ----------------------------
# 1) How many datapoints (coverage)
# ----------------------------
n_total = len(df)
n_fews = df[FEWS].notna().sum()
n_ipcch = df[IPCCH].notna().sum()
n_both = (df[FEWS].notna() & df[IPCCH].notna()).sum()

print("\n=== Coverage (number of datapoints) ===")
print(f"Total rows: {n_total}")
print(f"FEWS non-missing:  {n_fews} ({n_fews/n_total:.1%})")
print(f"IPCCH non-missing: {n_ipcch} ({n_ipcch/n_total:.1%})")
print(f"Overlap (both):    {n_both} ({n_both/n_total:.1%})")

# ----------------------------
# 2) Average + spread (simple descriptive stats)
# ----------------------------
def simple_stats(s):
    s = s.dropna()
    return {
        "n": int(len(s)),
        "mean": float(s.mean()) if len(s) else np.nan,
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "min": float(s.min()) if len(s) else np.nan,
        "25%": float(s.quantile(0.25)) if len(s) else np.nan,
        "50%": float(s.median()) if len(s) else np.nan,
        "75%": float(s.quantile(0.75)) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
        "unique_values": int(s.nunique())
    }

print("\n=== Descriptives (all available rows) ===")
print("FEWS :", simple_stats(df[FEWS]))
print("IPCCH:", simple_stats(df[IPCCH]))

# Overlap dataset
both = df[[TIME, A1, A2, FEWS, IPCCH]].dropna(subset=[FEWS, IPCCH]).copy()
both[FEWS] = both[FEWS].round().astype(int)
both[IPCCH] = both[IPCCH].round().astype(int)

print("\n=== Descriptives (overlap only) ===")
print("FEWS overlap :", simple_stats(both[FEWS]))
print("IPCCH overlap:", simple_stats(both[IPCCH]))

# ----------------------------
# 3) Agreement / disagreement (easy to interpret)
# ----------------------------
both["diff"] = both[FEWS] - both[IPCCH]
both["abs_diff"] = both["diff"].abs()
both["disagree"] = (both[FEWS] != both[IPCCH])

exact_agree = 1 - both["disagree"].mean()
avg_abs_diff = both["abs_diff"].mean()
share_within1 = (both["abs_diff"] <= 1).mean()

print("\n=== Agreement between FEWS and IPCCH (overlap rows) ===")
print(f"Exact agreement rate (same phase): {exact_agree:.1%}")
print(f"Average absolute difference (how many phases apart on avg): {avg_abs_diff:.3f}")
print(f"Share within 1 phase difference: {share_within1:.1%}")

print("\nDifference counts (FEWS - IPCCH):")
print(both["diff"].value_counts().sort_index())

# ----------------------------
# 4) Crisis vs non-crisis agreement (very interpretable)
# Crisis = phase >= 3
# ----------------------------
both["fews_crisis"] = both[FEWS] >= 3
both["ipcch_crisis"] = both[IPCCH] >= 3
crisis_agree = (both["fews_crisis"] == both["ipcch_crisis"]).mean()

print("\n=== Crisis agreement (Phase >= 3) ===")
print(f"Crisis/non-crisis agreement rate: {crisis_agree:.1%}")

# ----------------------------
# 5) ADMIN1 / ADMIN2 disagreement counts
# "How many different admin1/admin2 have disagreement?"
# ----------------------------
print("\n=== Admin coverage in overlap ===")
print("Unique ADMIN1 in overlap:", both[A1].nunique())
print("Unique ADMIN2 in overlap:", both[A2].nunique())

admin1_any_disagree = both.groupby(A1)["disagree"].any()
admin2_any_disagree = both.groupby(A2)["disagree"].any()

print("\n=== How many admin units show ANY disagreement? ===")
print("ADMIN1 with any disagreement:", int(admin1_any_disagree.sum()), "/", both[A1].nunique())
print("ADMIN2 with any disagreement:", int(admin2_any_disagree.sum()), "/", both[A2].nunique())

# Show the worst admin units (highest disagreement rate), but only among units with enough rows
def top_disagreement_units(unit_col, min_rows=10, top_n=15):
    g = both.groupby(unit_col).agg(
        n=("disagree", "size"),
        disagree_rate=("disagree", "mean"),
        mean_abs_diff=("abs_diff", "mean"),
        max_abs_diff=("abs_diff", "max")
    )
    g = g[g["n"] >= min_rows].sort_values("disagree_rate", ascending=False)
    return g.head(top_n)

print("\n=== Top ADMIN1 disagreement (min 10 overlap rows) ===")
print(top_disagreement_units(A1, min_rows=10, top_n=15).to_string())

print("\n=== Top ADMIN2 disagreement (min 10 overlap rows) ===")
print(top_disagreement_units(A2, min_rows=10, top_n=15).to_string())

# ----------------------------
# 6) Optional: disagreement over time (helps interpret if it happens in certain periods)
# ----------------------------
time_summary = both.groupby(TIME).agg(
    n=("disagree", "size"),
    disagree_rate=("disagree", "mean"),
    avg_abs_diff=("abs_diff", "mean")
).sort_values("disagree_rate", ascending=False)

print("\n=== Months with highest disagreement (top 12) ===")
print(time_summary.head(12).to_string())
