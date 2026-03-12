import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from xgboost.callback import EarlyStopping




# -----------------------------
# Load data
# -----------------------------
#DATA_PATH = r"C:\Users\meesw\projects\Thesis\DATA\admin1_dataset_spatial_small.parquet"
DATA_PATH = r"C:\Users\meesw\projects\Thesis\DATA\admin1_dataset_v2.parquet"

df = pd.read_parquet(DATA_PATH)

df.drop(columns=["year", "ipc_phase_fews_lag3", "ipc_phase_fews_lag1"], inplace=True)       #Remove year to not see tempral pattern so much

df["year_month"] = pd.to_datetime(df["year_month"])
df = df.sort_values(["year_month", "ADMIN1"]).reset_index(drop=True)

TARGET = "ipc_phase_fews"
ID_COLS_ADMIN1 = ["year_month", "ADMIN0", "ADMIN1", TARGET]

FEATURES = [c for c in df.columns if c not in ID_COLS_ADMIN1]

X = df[FEATURES]
y = df[TARGET]


# -----------------------------
# Time-based split (same style as regression)
# last N months as test set
# -----------------------------
TEST_MONTHS = 6
unique_months = np.sort(df["year_month"].unique())
if len(unique_months) <= TEST_MONTHS:
    raise ValueError(f"Not enough months ({len(unique_months)}) for TEST_MONTHS={TEST_MONTHS}")

test_months = unique_months[-TEST_MONTHS:]
train_mask = ~df["year_month"].isin(test_months)
test_mask = df["year_month"].isin(test_months)

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

y_train = y_train.clip(upper=3)
y_test  = y_test.clip(upper=3)                  # Clipped the target variable to max 3 because there is only one 4

# map to 0..3
y_train = (y_train - 1).astype(int)
y_test  = (y_test - 1).astype(int)

print(f"Train rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")
print(f"Train months: {df.loc[train_mask, 'year_month'].nunique()}, Test months: {df.loc[test_mask, 'year_month'].nunique()}")
print(f"Num features: {len(FEATURES)}")

print("Unique y_train raw:", sorted(y.loc[train_mask].dropna().unique()))
print("Unique y_test raw :", sorted(y.loc[test_mask].dropna().unique()))

print("Unique y_train mapped:", sorted(y_train.unique()))
print("Unique y_test mapped :", sorted(y_test.unique()))
print("Max y_train mapped:", y_train.max(), "Max y_test mapped:", y_test.max())


#add weights

# Get unique classes in training
classes = np.unique(y_train)

# Compute balanced weights
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# Map class → weight
class_weight_dict = dict(zip(classes, weights))

print("Class weights:", class_weight_dict)

# Create sample weight vector
sample_weights = y_train.map(class_weight_dict)

# -----------------------------
# XGBoost Regressor
# -----------------------------
# These settings are conservative for small-ish tabular datasets.
# You can tune n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
model = XGBClassifier(
    n_estimators=210,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
#    early_stopping=True


)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,   
    verbose=30,
    eval_set=[(X_test, y_test)]
)


# -----------------------------
# Evaluate
# -----------------------------
pred = model.predict(X_test)




#r2 = r2_score(y_test, pred)
#mae = mean_absolute_error(y_test, pred)
#rmse = np.sqrt(mean_squared_error(y_test, pred)) FOR REGRESSION

print("\n--- XGBoost Baseline ---")
best_iter = getattr(model, "best_iteration", None)
if best_iter is not None:
    print(f"Best trees (best_iteration): {best_iter}")
else:
    print(f"Trees used (n_estimators): {model.n_estimators}")
#print(f"R²   : {r2:.4f}")
#print(f"MAE  : {mae:.4f}")
#print(f"RMSE : {rmse:.4f}") FOR REGRESSION

print("Accuracy:", accuracy_score(y_test, pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))


# -----------------------------
# Feature importance (gain-based)
# -----------------------------
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("\n--- Top 20 feature importances ---")
print(importances.head(20))


# -----------------------------
# Save predictions for inspection (optional)
# -----------------------------
out = df.loc[test_mask, ["year_month", "ADMIN0", "ADMIN1", TARGET]].copy()
out["pred_xgb"] = pred
out_path = r"C:\Users\meesw\projects\Thesis\DATA\xgb_predictions.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved test predictions to: {out_path}")