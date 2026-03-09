import pandas as pd
import os

in_path = r"DATA\v3-hfid-drivers-dataset.csv"
out_dir = r"DATA"
os.makedirs(out_dir, exist_ok=True)

chunksize = 500_000
i = 0

for chunk in pd.read_csv(in_path, chunksize=chunksize, low_memory=False):
    af = chunk[chunk["ADMIN0"] == "Afghanistan"]
    if not af.empty:
        af.to_parquet(f"{out_dir}/part_{i:04d}.parquet", index=False)
        i += 1

print("Wrote parts:", i, "in", out_dir)
