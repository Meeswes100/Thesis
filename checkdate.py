import pandas as pd

df = pd.read_parquet(r"DATA\part_0000.parquet")

# Make sure it's datetime
df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")

# Extract day
df["day"] = df["year_month"].dt.day

print("Unique days in dataset:")
print(sorted(df["day"].dropna().unique()))

print("\nDay distribution:")
print(df["day"].value_counts().sort_index())
