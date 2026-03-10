import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\meesw\projects\Thesis\DATA\admin1_dataset_v2.parquet"
OUTPUT_DIR = r"C:\Users\meesw\projects\Thesis\EDA_admin1"


df = pd.read_parquet(DATA_PATH)

print(df.head())

df_year_ipc_avg = df.groupby(["year"]).["ipc_phase_fews"].mean().head()
df_year_ipc_min = df.groupby(["year"])["ipc_phase_fews"].min().head()
df_year_ipc_max = df.groupby(["year"])["ipc_phase_fews"].max().head()




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