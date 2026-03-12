import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------
# Load data
# -----------------------------
DATA_PATH = r"C:\Users\meesw\projects\Thesis\DATA\admin1_dataset_v2.parquet"
df = pd.read_parquet(DATA_PATH)

# Ensure correct dtypes / ordering
df["year_month"] = pd.to_datetime(df["year_month"])
df = df.sort_values(["year_month", "ADMIN1"]).reset_index(drop=True)

TARGET = "ipc_phase_fews"
ID_COLS_ADMIN1 = ["year_month", "ADMIN0", "ADMIN1", TARGET]  # ids + target

# Remove to test presistence
REMOVE_FEATURES = [
    "year",
    "ipc_phase_fews_lag1",
    "ipc_phase_fews_lag3"
]

FEATURES = [c for c in df.columns if c not in ID_COLS_ADMIN1 + REMOVE_FEATURES]

X = df[FEATURES]
y = df[TARGET]

TEST_MONTHS = 6

unique_months = np.sort(df["year_month"].unique())
if len(unique_months) <= TEST_MONTHS:
    raise ValueError(f"Not enough months ({len(unique_months)}) for TEST_MONTHS={TEST_MONTHS}")

test_months = unique_months[-TEST_MONTHS:]
train_mask = ~df["year_month"].isin(test_months)
test_mask = df["year_month"].isin(test_months)

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

print(f"Train rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")
print(f"Train months: {df.loc[train_mask, 'year_month'].nunique()}, Test months: {df.loc[test_mask, 'year_month'].nunique()}")
print(f"Num features: {len(FEATURES)}")

ALPHA = 1.0  # regularization strength; try 0.1, 1, 10
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=ALPHA, random_state=42))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n--- Ridge Regression Baseline ---")
print(f"alpha: {ALPHA}")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


# -----------------------------
# Inspect coefficients (most positive / most negative)
# Note: because we scale features, coefficients are comparable
# -----------------------------
coefs = pd.Series(model.named_steps["ridge"].coef_, index=FEATURES).sort_values()

print("\n--- Top negative coefficients ---")
print(coefs.head(15))

print("\n--- Top positive coefficients ---")
print(coefs.tail(15))


# -----------------------------
# Save predictions for inspection (optional)
# -----------------------------
out = df.loc[test_mask, ["year_month", "ADMIN0", "ADMIN1", TARGET]].copy()
out["pred_ridge"] = pred
out_path = r"DATA/ridge_predictions.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved test predictions to: {out_path}")