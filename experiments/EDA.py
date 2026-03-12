import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# SETTINGS
# =========================================================
DATA_PATH = r"C:\Users\meesw\projects\Thesis\DATA\admin1_dataset_v2.parquet"
OUTPUT_DIR = r"C:\Users\meesw\projects\Thesis\EDA_admin1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_parquet(DATA_PATH)

df["year_month"] = pd.to_datetime(df["year_month"])

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nDTYPES:")
print(df.dtypes)

# =========================================================
# BASIC DATA CHECKS
# =========================================================
print("\n" + "=" * 60)
print("BASIC CHECKS")
print("=" * 60)

print("\nNumber of unique countries:", df["ADMIN0"].nunique())
print("Number of unique ADMIN1 regions:", df["ADMIN1"].nunique())
print("Date range:", df["year_month"].min(), "to", df["year_month"].max())

duplicates = df[["year_month", "ADMIN0", "ADMIN1"]].duplicated().sum()
print("Duplicate ADMIN1-month rows:", duplicates)

print("\nRows per ADMIN1 (number of months):")
print(df.groupby("ADMIN1")["year_month"].nunique().describe())

# =========================================================
# MISSING VALUES
# =========================================================


# =========================================================
# TARGET DISTRIBUTION
# =========================================================
print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_dist = df["ipc_phase_fews"].value_counts(dropna=False).sort_index()
target_dist_norm = df["ipc_phase_fews"].value_counts(normalize=True, dropna=False).sort_index()

print("\nCounts:")
print(target_dist)

print("\nProportions:")
print(target_dist_norm)

target_dist.to_csv(os.path.join(OUTPUT_DIR, "ipc_distribution_counts.csv"))
target_dist_norm.to_csv(os.path.join(OUTPUT_DIR, "ipc_distribution_proportions.csv"))

plt.figure(figsize=(7, 5))
df["ipc_phase_fews"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution of IPC phase")
plt.xlabel("IPC phase")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ipc_distribution_bar.png"), dpi=300)
plt.close()

# =========================================================
# IPC OVER TIME
# =========================================================
print("\n" + "=" * 60)
print("IPC OVER TIME")
print("=" * 60)

ipc_over_time = df.groupby("year_month")["ipc_phase_fews"].agg(["mean", "median", "std", "count"])
print(ipc_over_time.head())

ipc_over_time.to_csv(os.path.join(OUTPUT_DIR, "ipc_over_time.csv"))

plt.figure(figsize=(12, 5))
plt.plot(ipc_over_time.index, ipc_over_time["mean"])
plt.title("Average IPC over time")
plt.xlabel("Date")
plt.ylabel("Average IPC")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ipc_over_time_mean.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(ipc_over_time.index, ipc_over_time["median"])
plt.title("Median IPC over time")
plt.xlabel("Date")
plt.ylabel("Median IPC")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ipc_over_time_median.png"), dpi=300)
plt.close()

# =========================================================
# SEASONALITY
# =========================================================
print("\n" + "=" * 60)
print("SEASONALITY")
print("=" * 60)

df["month"] = df["year_month"].dt.month
df["year"] = df["year_month"].dt.year

seasonality = df.groupby("month")["ipc_phase_fews"].agg(["mean", "median", "count"])
print(seasonality)

seasonality.to_csv(os.path.join(OUTPUT_DIR, "ipc_seasonality.csv"))

plt.figure(figsize=(8, 5))
plt.plot(seasonality.index, seasonality["mean"], marker="o")
plt.title("Seasonality of IPC")
plt.xlabel("Month")
plt.ylabel("Average IPC")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ipc_seasonality_mean.png"), dpi=300)
plt.close()

# =========================================================
# REGIONAL DIFFERENCES
# =========================================================
print("\n" + "=" * 60)
print("REGIONAL DIFFERENCES")
print("=" * 60)

region_mean = df.groupby("ADMIN1")["ipc_phase_fews"].mean().sort_values(ascending=False)
print(region_mean.head(20))

region_mean.to_csv(os.path.join(OUTPUT_DIR, "admin1_avg_ipc.csv"))

plt.figure(figsize=(12, max(6, len(region_mean) * 0.25)))
region_mean.sort_values().plot(kind="barh")
plt.title("Average IPC by ADMIN1")
plt.xlabel("Average IPC")
plt.ylabel("ADMIN1")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "admin1_avg_ipc_barh.png"), dpi=300)
plt.close()

# =========================================================
# TOP / BOTTOM REGIONS
# =========================================================
top10 = region_mean.head(10)
bottom10 = region_mean.tail(10)

print("\nTop 10 highest average IPC ADMIN1 regions:")
print(top10)

print("\nTop 10 lowest average IPC ADMIN1 regions:")
print(bottom10)

top10.to_csv(os.path.join(OUTPUT_DIR, "top10_admin1_ipc.csv"))
bottom10.to_csv(os.path.join(OUTPUT_DIR, "bottom10_admin1_ipc.csv"))

# =========================================================
# DATA AVAILABILITY OVER TIME
# =========================================================
print("\n" + "=" * 60)
print("DATA AVAILABILITY OVER TIME")
print("=" * 60)

rows_per_year = df.groupby("year").size()
print(rows_per_year)

rows_per_year.to_csv(os.path.join(OUTPUT_DIR, "rows_per_year.csv"))

plt.figure(figsize=(8, 5))
rows_per_year.plot(kind="bar")
plt.title("Number of observations per year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rows_per_year.png"), dpi=300)
plt.close()

# =========================================================
# FEATURE LIST
# =========================================================
ID_COLS = ["year_month", "ADMIN0", "ADMIN1", "ipc_phase_fews"]

feature_cols = [c for c in df.columns if c not in ID_COLS]

# Keep only numeric features for EDA
numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print("\n" + "=" * 60)
print("NUMERIC FEATURES")
print("=" * 60)
print(numeric_feature_cols)

# =========================================================
# CORRELATION WITH TARGET
# =========================================================
print("\n" + "=" * 60)
print("SPEARMAN CORRELATION WITH IPC")
print("=" * 60)

corr_with_target = (
    df[numeric_feature_cols + ["ipc_phase_fews"]]
    .corr(method="spearman")["ipc_phase_fews"]
    .drop("ipc_phase_fews")
    .sort_values(key=lambda x: x.abs(), ascending=False)
)

print(corr_with_target)

corr_with_target.to_csv(os.path.join(OUTPUT_DIR, "spearman_corr_with_ipc.csv"))

plt.figure(figsize=(10, max(6, len(corr_with_target) * 0.3)))
corr_with_target.sort_values().plot(kind="barh")
plt.title("Spearman correlation of features with IPC")
plt.xlabel("Correlation with IPC")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "corr_with_ipc_barh.png"), dpi=300)
plt.close()

# =========================================================
# FULL CORRELATION HEATMAP
# =========================================================
print("\n" + "=" * 60)
print("FULL CORRELATION HEATMAP")
print("=" * 60)

# Keep it readable: only numeric columns
corr_matrix = df[numeric_feature_cols + ["ipc_phase_fews"]].corr(method="spearman")
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "full_spearman_corr_matrix.csv"))

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Spearman correlation heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()

# =========================================================
# HIGH CORRELATIONS BETWEEN FEATURES
# =========================================================
print("\n" + "=" * 60)
print("HIGH FEATURE-FEATURE CORRELATIONS")
print("=" * 60)

corr_abs = corr_matrix.abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

high_corr_pairs = (
    upper.stack()
    .sort_values(ascending=False)
    .reset_index()
)

high_corr_pairs.columns = ["feature_1", "feature_2", "abs_spearman_corr"]

print(high_corr_pairs.head(30))
high_corr_pairs.to_csv(os.path.join(OUTPUT_DIR, "high_corr_pairs.csv"), index=False)

# =========================================================
# FEATURE DISTRIBUTIONS
# =========================================================
print("\n" + "=" * 60)
print("FEATURE DISTRIBUTIONS")
print("=" * 60)

dist_dir = os.path.join(OUTPUT_DIR, "feature_distributions")
os.makedirs(dist_dir, exist_ok=True)

for col in numeric_feature_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    safe_name = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(os.path.join(dist_dir, f"{safe_name}.png"), dpi=300)
    plt.close()

# =========================================================
# BOXPLOTS OF FEATURES
# =========================================================
boxplot_dir = os.path.join(OUTPUT_DIR, "feature_boxplots")
os.makedirs(boxplot_dir, exist_ok=True)

for col in numeric_feature_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    safe_name = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(os.path.join(boxplot_dir, f"{safe_name}.png"), dpi=300)
    plt.close()

# =========================================================
# RELATIONSHIPS WITH TARGET (SCATTER / BOXPLOT)
# =========================================================
print("\n" + "=" * 60)
print("FEATURES VS TARGET")
print("=" * 60)

rel_dir = os.path.join(OUTPUT_DIR, "feature_vs_target")
os.makedirs(rel_dir, exist_ok=True)

for col in numeric_feature_cols:
    # Scatter
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x=col, y="ipc_phase_fews", alpha=0.5)
    plt.title(f"{col} vs IPC")
    plt.tight_layout()
    safe_name = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(os.path.join(rel_dir, f"{safe_name}_scatter.png"), dpi=300)
    plt.close()

    # Boxplot by IPC phase
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="ipc_phase_fews", y=col)
    plt.title(f"{col} by IPC phase")
    plt.tight_layout()
    plt.savefig(os.path.join(rel_dir, f"{safe_name}_by_ipc_boxplot.png"), dpi=300)
    plt.close()

# =========================================================
# LAGGED IPC ANALYSIS
# =========================================================
print("\n" + "=" * 60)
print("LAGGED IPC ANALYSIS")
print("=" * 60)

lag_cols = [c for c in df.columns if "lag" in c.lower() and c in df.select_dtypes(include=[np.number]).columns]

if "ipc_phase_fews_lag1" in df.columns:
    corr_lag1 = df[["ipc_phase_fews", "ipc_phase_fews_lag1"]].corr(method="spearman").iloc[0, 1]
    print("Spearman corr IPC vs IPC_lag1:", corr_lag1)

if "ipc_phase_fews_lag3" in df.columns:
    corr_lag3 = df[["ipc_phase_fews", "ipc_phase_fews_lag3"]].corr(method="spearman").iloc[0, 1]
    print("Spearman corr IPC vs IPC_lag3:", corr_lag3)

# =========================================================
# COUNTRY-LEVEL AVERAGE IPC OVER TIME
# =========================================================
print("\n" + "=" * 60)
print("COUNTRY-LEVEL IPC TRENDS")
print("=" * 60)

country_time = df.groupby(["year_month", "ADMIN0"])["ipc_phase_fews"].mean().reset_index()
country_time.to_csv(os.path.join(OUTPUT_DIR, "country_ipc_over_time.csv"), index=False)

for country in country_time["ADMIN0"].dropna().unique():
    tmp = country_time[country_time["ADMIN0"] == country].sort_values("year_month")

    plt.figure(figsize=(10, 4))
    plt.plot(tmp["year_month"], tmp["ipc_phase_fews"])
    plt.title(f"Average IPC over time - {country}")
    plt.xlabel("Date")
    plt.ylabel("Average IPC")
    plt.tight_layout()

    safe_country = str(country).replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"ipc_over_time_{safe_country}.png"), dpi=300)
    plt.close()

# =========================================================
# SUMMARY TABLE FOR THESIS
# =========================================================
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

summary_table = pd.DataFrame({
    "n_rows": [len(df)],
    "n_countries": [df["ADMIN0"].nunique()],
    "n_admin1": [df["ADMIN1"].nunique()],
    "start_date": [df["year_month"].min()],
    "end_date": [df["year_month"].max()],
    "mean_ipc": [df["ipc_phase_fews"].mean()],
    "median_ipc": [df["ipc_phase_fews"].median()],
    "std_ipc": [df["ipc_phase_fews"].std()]
})

print(summary_table)
summary_table.to_csv(os.path.join(OUTPUT_DIR, "dataset_summary_table.csv"), index=False)

# =========================================================
# OPTIONAL: NICE SMALL TABLE OF TOP CORRELATES
# =========================================================
top_pos = corr_with_target.sort_values(ascending=False).head(10)
top_neg = corr_with_target.sort_values(ascending=True).head(10)

print("\nTop positive correlations with IPC:")
print(top_pos)

print("\nTop negative correlations with IPC:")
print(top_neg)

top_pos.to_csv(os.path.join(OUTPUT_DIR, "top_positive_corr_with_ipc.csv"))
top_neg.to_csv(os.path.join(OUTPUT_DIR, "top_negative_corr_with_ipc.csv"))

print("\n" + "=" * 60)
print("EDA FINISHED")
print("Results saved in:", OUTPUT_DIR)
print("=" * 60)