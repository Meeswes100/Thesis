import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = r"C:\Users\meesw\projects\Thesis\DATA\admin1_dataset_v2.parquet"
OUTPUT_DIR = r"C:\Users\meesw\projects\Thesis\EDA_admin1"


df = pd.read_parquet(DATA_PATH)

print(df.head())

df_year_ipc_avg = df.groupby(["year"])["ipc_phase_fews"].mean()
df_year_ipc_min = df.groupby(["year"])["ipc_phase_fews"].min()
df_year_ipc_max = df.groupby(["year"])["ipc_phase_fews"].max()


df_year_ipc = df.groupby(["year"]).agg({"ipc_phase_fews": ["mean", "std"]})
print(df_year_ipc.head())

plt.figure()
df_year_ipc_avg.plot()
plt.title("IPC phase fews mean vs time ")
plt.xlabel("year")
plt.ylabel("IPC phase fews")
plt.show()

plt.figure()
df_year_ipc_min.plot()
plt.title("IPC phase fews min vs time ")
plt.xlabel("year")
plt.ylabel("IPC phase fews")
plt.show()

plt.figure()
df_year_ipc_max.plot()
plt.title("IPC phase fews min vs time ")
plt.xlabel("year")
plt.ylabel("IPC phase fews")
plt.show()

plt.figure()
mean = df_year_ipc["ipc_phase_fews"]["mean"]
std = df_year_ipc["ipc_phase_fews"]["std"]
plt.plot(mean.index, mean.values, label="Mean IPC")
plt.fill_between(
    mean.index,
    mean - std,
    mean + std,
    alpha=0.3,
    label="±1 Std Dev"
)
plt.title("Average IPC Phase with Standard Deviation")
plt.xlabel("Year")
plt.ylabel("IPC Phase")
plt.legend()

plt.show()

#SEASONALITY
df["year_month"] = pd.to_datetime(df["year_month"])

df["month"] = df["year_month"].dt.month

monthly_ipc = df.groupby("month")["ipc_phase_fews"].mean()

monthly_std = df.groupby("month")["ipc_phase_fews"].std()

# Plot
plt.figure(figsize=(8,5))

plt.plot(monthly_ipc.index, monthly_ipc.values, marker="o", label="Mean IPC")

plt.fill_between(
    monthly_ipc.index,
    monthly_ipc - monthly_std,
    monthly_ipc + monthly_std,
    alpha=0.2,
    label="±1 Std Dev"
)

plt.xticks(range(1,13))
plt.xlabel("Month")
plt.ylabel("Average IPC Phase")
plt.title("Seasonality of IPC Phase (Average per Month)")
plt.legend()

plt.tight_layout()
plt.show()


