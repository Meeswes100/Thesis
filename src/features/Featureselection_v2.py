import pandas as pd
import numpy as np


ID_COLS = ["year_month", "ADMIN0", "ADMIN1", "ADMIN2", "ipc_phase_fews"]
ID_COLS_ADMIN1 = ["year_month", "ADMIN0", "ADMIN1", "ipc_phase_fews"]


FEATURES_BASELINE = [

    # food scores
#    "fcs_lit",          #proportion of population with poor/borderline food consumption.    # litrature is more stable but less frequent
#    "rcsi_lit",         #Proportion of population using severe coping strategies. 
#    "fcs_rt mean",      #more frequent less stable
#    "rcsi_rt mean",    Almost no data avainable

    # Agroclimatology (weather/vegetation/drought) 
    "mean_SPI3_crop",  #short term drought
    "mean_RAIN_mean_rolling_sum_3m_crop",   
    "mean_RAIN_mean_rolling_sum_12m_crop",
    "mean_SM_combined_mean_rolling_sum_3m_crop",
    "mean_FPAR_mean_zscore_crop",  #how much photosyntetis in the plant. Better the NDVI zscore is less or more than normal
    "mean_TEMP_mean_zscore_crop",   

    #  Conflict 
#    "acled_events", # how many violent incidents                   DROPPED BECAUSE DATA NOT AVAINABLE 1
#    "acled_fatalities", # number of deaths from conflicts          DROPPED BECAUSE DATA NOT AVAINABLE 1
    "ucdp_events",  # another dataset coutning conflict
    "ucdp_deaths",  # another dataset counting deaths
    # TODO maybe make a rolling mean

    # Displacement 
#    "idps", # people who had to leave home                          MISSING 0.707 SHOULD I NOT USE AT AL?
#    "idps_3m_sum", #                                                 MISSING 0.707 SHOULD I NOT USE AT AL?
    "Conflict Internal Displacements",
    "Conflict Stock Displacement",
    "Disaster Internal Displacements",
#    "Disaster Stock Displacment", # probually choise one of the two depends on if i used lagged ipc MISSING 0.7

    # Prices / economic access 
    "Food Price Index",
    "Cereals Price Index",
    "Oils Price Index",
    "Sugar Price Index",
    # maybe z score or variation of these is better?

    # Structural vulnerability
    "INFORM Risk Index", # how vunraible is the area
    "GDP_annual_growth_perc_filled",    #is it going better or worse with the area
    "Tot_Pop",  #total pop
    "Rural_Pop",    # rural pop TODO should probually make a precentage or factor of these

    # AID
    "ha_fews",

    # TODO should add month for seasonality

]



df = pd.read_parquet(r"C:\Users\meesw\projects\Thesis\DATA\part_0000.parquet")

df = df[ID_COLS + FEATURES_BASELINE]

print("Missing before removing non ipc values")

print(df.isna().mean().sort_values())



df = df[df["ipc_phase_fews"].notna()]

print("Distrobution of ipc fews before aggegating", df["ipc_phase_fews"].value_counts(normalize=True))





print(df.head())

print(df.isna().mean().sort_values())



df["year_month"] = pd.to_datetime(df["year_month"])

#check for duplicates

print("amount of duplicates in data", df[["ADMIN0", "ADMIN1", "ADMIN2", "year_month"]].duplicated().sum())

#check for inbalances 
print("Imbalances Admin2")
print("balance in admin2", df.groupby("ADMIN2")["year_month"].nunique().describe())

#how much variation in admin1

print("variations in admin1", df.groupby(["ADMIN1", "year_month"])["ipc_phase_fews"].std().describe())

#when is inform missing

#print("when is inform missing", df.groupby("year")["INFORM Risk Index"].mean()) REMOVVED YEAR

#check if mean, median or population based mean is best for ipc aggragation

ipc_mean = df.groupby(["year_month", "ADMIN1"])["ipc_phase_fews"].mean().reset_index(name = "ipc_mean")
ipc_median = df.groupby(["year_month", "ADMIN1"])["ipc_phase_fews"].median().reset_index(name = "ipc_median")

ipc_weighted = df.groupby(["ADMIN1", "year_month"]).apply(lambda x: np.average(x["ipc_phase_fews"], weights=x["Tot_Pop"])).reset_index(name = "ipc_weighted")
                                                          
ipc_compare = (
    ipc_mean
    .merge(ipc_median, on=["ADMIN1", "year_month"])
    .merge(ipc_weighted, on=["ADMIN1", "year_month"])
)

ipc_compare["mean_minus_median"] = (
    ipc_compare["ipc_mean"] - ipc_compare["ipc_median"]
)

ipc_compare["weighted_minus_mean"] = (
    ipc_compare["ipc_weighted"] - ipc_compare["ipc_mean"]
)

ipc_compare["weighted_minus_median"] = (
    ipc_compare["ipc_weighted"] - ipc_compare["ipc_median"]
)
print(  
ipc_compare[
    ["mean_minus_median",
     "weighted_minus_mean",
     "weighted_minus_median"]
].describe()
)

#aggegating 

GROUP_COLS = ["year_month", "ADMIN0", "ADMIN1"]

SUM_COLS = [
    "ucdp_events",
    "ucdp_deaths",
    "Conflict Internal Displacements",
    "Conflict Stock Displacement",
    "Disaster Internal Displacements",
    "Tot_Pop",
    "Rural_Pop",
]

ALL_FEATURES = [c for c in FEATURES_BASELINE if c not in ID_COLS]
MEDIAN_COLS = [c for c in ALL_FEATURES if c not in SUM_COLS and c != "ipc_phase_fews"]

agg_dict = {c: "sum" for c in SUM_COLS}
agg_dict.update({c: "median" for c in MEDIAN_COLS})
agg_dict["ipc_phase_fews"] = lambda x: x.mode().iloc[0]

df_admin1 = (
    df.groupby(GROUP_COLS, as_index=False)
      .agg(agg_dict)
)

#how many datapoint per admin1 now
print("datapoint per admin1", df_admin1.groupby("ADMIN1")["year_month"].nunique().describe())

#imputate
print("missing", df_admin1.isna().mean())

#inform is de only one thats missing so its imputated

col = "INFORM Risk Index"

df_admin1[col + "_missing"] = df_admin1[col].isna().astype(int)
df_admin1[col] = df_admin1[col].fillna(df_admin1[col].median())

print("missing after imputation", df_admin1.isna().mean())

#feature enginering

ENGINEERED_FEATURES = [

    

    # Structural
    "rural_share",

    # Seasonality
    "month_sin",
    "month_cos",

    # Missingness
    "INFORM Risk Index_missing",

    # Price dynamics
    "Food Price Index_z",
    "Cereals Price Index_z",
    "Oils Price Index_z",
    "Sugar Price Index_z",

    "Food Price Index_change",
    "Cereals Price Index_change",
    "Oils Price Index_change",
    "Sugar Price Index_change",

    # Climate lags
    "mean_SPI3_crop_lag1",
    "mean_SPI3_crop_lag3",

    "mean_FPAR_mean_zscore_crop_lag1",
    "mean_FPAR_mean_zscore_crop_lag3",

    "mean_SM_combined_mean_rolling_sum_3m_crop_lag1",
    "mean_SM_combined_mean_rolling_sum_3m_crop_lag3",

    # Shocks
    "drought_shock",
    "veg_shock",

    # Conflict
    "conflict_3m"
]

df_admin1["rural_share"] = df_admin1["Rural_Pop"] / df_admin1["Tot_Pop"].replace(0, np.nan)
df_admin1["month"] = df_admin1["year_month"].dt.month
df_admin1["year"] = df_admin1["year_month"].dt.year

df_admin1["month_sin"] = np.sin(2*np.pi*df_admin1["month"]/12)
df_admin1["month_cos"] = np.cos(2*np.pi*df_admin1["month"]/12)
df_admin1.drop(columns=["month"], inplace=True)

PRICE_COLS = [
    "Food Price Index",
    "Cereals Price Index",
#    "Oils Price Index",
#    "Sugar Price Index"
]

    #relative price

for col in PRICE_COLS:
    df_admin1[col + "_z"] = (
        df_admin1.groupby("ADMIN1")[col]
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    #price change

for col in PRICE_COLS:
    df_admin1[col + "_change"] = (
        df_admin1.groupby("ADMIN1")[col]
        .diff()
    )

df_admin1.drop(columns=PRICE_COLS, inplace=True)


    #lagged features

CLIMATE_COLS = [
    "ipc_phase_fews",                                 #Needed lag for this as well maybe not in the right place
    "mean_SPI3_crop",
    "mean_FPAR_mean_zscore_crop",
    "mean_SM_combined_mean_rolling_sum_3m_crop"
]

for col in CLIMATE_COLS:
    df_admin1[col + "_lag1"] = df_admin1.groupby("ADMIN1")[col].shift(1)
    df_admin1[col + "_lag3"] = df_admin1.groupby("ADMIN1")[col].shift(3)

    #shock indicators



df_admin1["drought_shock"] = (df_admin1["mean_SPI3_crop"] < -1).astype(int)
df_admin1["veg_shock"] = (df_admin1["mean_FPAR_mean_zscore_crop"] < -1).astype(int)

    #conlfict rolling

df_admin1["conflict_3m"] = (
    df_admin1.groupby("ADMIN1")["ucdp_events"]
    .rolling(3).sum().reset_index(level=0, drop=True)
)



#correlation

FEATURES_MODEL = (
    df_admin1
        .drop(columns=ID_COLS_ADMIN1)   # remove identifiers
        .columns
        .tolist()
)

corr = df_admin1[FEATURES_MODEL].corr(method="spearman")

high_corr = (
    corr.abs()
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
)

print("correlation")
print(high_corr.head(20))

#drop things with to high correlation

df_admin1.drop(columns=["ucdp_deaths"], inplace=True)       #we use events instead is more stable and highly correlated
df_admin1.drop(columns=["mean_RAIN_mean_rolling_sum_3m_crop"], inplace=True)       #instaed we have soil moister highly correlated
df_admin1.drop(columns=["Oils Price Index"], inplace=True)
df_admin1.drop(columns=["Sugar Price Index"], inplace=True) #was to high correlation between all 4 so i removed these to. If i add add z scores and change again

FEATURES_MODEL = (
    df_admin1
        .drop(columns=ID_COLS_ADMIN1)   # remove identifiers
        .columns
        .tolist()
)

corr = df_admin1[FEATURES_MODEL].corr(method="spearman")

high_corr = (
    corr.abs()
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
)

print("correlation AFTER REMOVING")
print(high_corr.head(20))

print("amount missing now", df_admin1.isna().sum())

#imputate the missing variables because of lagged with median

for col in FEATURES_MODEL:
    df_admin1[col] = (
        df_admin1.groupby("ADMIN1")[col]
        .transform(lambda x: x.fillna(x.median()))
    )

#check for constants

print(df_admin1[FEATURES_MODEL].std().sort_values().head())

print(df_admin1.head())

print("distorbution of target variable: ")

print(df_admin1["ipc_phase_fews"].value_counts(normalize=True))

df_admin1.to_parquet(r"C:\Users\meesw\projects\Thesis\DATA\admin1_dataset_v2.parquet", index=False)




















