"""Train day-ahead price and negative-price models; test on 2025."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, roc_auc_score

DATA = Path("data/processed/hourly.parquet")
OUT = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)
TEST_YEAR = 2025

df = pd.read_parquet(DATA)

# --- 1. Features known the day before delivery ------------------------------
# Day-ahead prices for day D are already known when bidding for D+1,
# so prices 24h and 168h earlier are legitimate inputs. Actual load and
# generation are NOT known in advance, so only forecasts are used.
df["price_lag_24"] = df["price"].shift(24)
df["price_lag_168"] = df["price"].shift(168)
daily_mean = df.groupby("date")["price"].mean()
df["prev_day_mean"] = df["date"].map(daily_mean.shift(1))

FEATURES = [
    "hour", "weekday", "month", "is_weekend", "is_holiday",
    "load_fc", "solar_fc", "wind_off_fc", "wind_on_fc", "wind_fc", "residual_load_fc",
    "price_lag_24", "price_lag_168", "prev_day_mean",
]

data = df.dropna(subset=["price"])
train = data[data["year"] < TEST_YEAR]
test = data[data["year"] == TEST_YEAR]
X_train, y_train = train[FEATURES], train["price"]
X_test, y_test = test[FEATURES], test["price"]
print(f"Train: {len(train):,} hours ({train.index.min():%Y-%m} to {train.index.max():%Y-%m})")
print(f"Test:  {len(test):,} hours ({TEST_YEAR})\n")

# --- 2. Price forecast model ------------------------------------------------
reg = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=42)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

mae = {
    "Naive: same hour yesterday": mean_absolute_error(y_test, test["price_lag_24"].fillna(y_train.mean())),
    "Naive: same hour last week": mean_absolute_error(y_test, test["price_lag_168"].fillna(y_train.mean())),
    "Gradient boosting model": mean_absolute_error(y_test, pred),
}
print("Price forecast error on 2025 (MAE, EUR/MWh, lower is better):")
for name, value in mae.items():
    print(f"  {name:<28} {value:6.2f}")
best_naive = min(mae["Naive: same hour yesterday"], mae["Naive: same hour last week"])
improvement = 1 - mae["Gradient boosting model"] / best_naive
print(f"  -> Model beats the best naive forecast by {improvement:.0%}\n")

# --- 3. Negative-price classifier --------------------------------------------
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=42)
clf.fit(X_train, train["is_negative"])
neg_prob = clf.predict_proba(X_test)[:, 1]
neg_pred = (neg_prob >= 0.5).astype(int)
cls_metrics = {
    "roc_auc": roc_auc_score(test["is_negative"], neg_prob),
    "precision": precision_score(test["is_negative"], neg_pred, zero_division=0),
    "recall": recall_score(test["is_negative"], neg_pred, zero_division=0),
}
print("Negative-price classifier on 2025:")
print(f"  ROC AUC   {cls_metrics['roc_auc']:.3f}  (1.0 = perfect, 0.5 = coin flip)")
print(f"  Precision {cls_metrics['precision']:.0%}  (of hours flagged negative, share that were)")
print(f"  Recall    {cls_metrics['recall']:.0%}  (of truly negative hours, share caught)\n")

# --- 4. What drives the forecast? -------------------------------------------
sample = X_test.sample(min(3000, len(X_test)), random_state=42)
imp = permutation_importance(reg, sample, y_test.loc[sample.index],
                             scoring="neg_mean_absolute_error", n_repeats=5, random_state=42)
importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)
print("Most important inputs (increase in MAE when shuffled, EUR/MWh):")
print(importance.round(2).head(8).to_string(), "\n")

# --- 5. Save everything the dashboard needs ---------------------------------
results = test[["price", "price_lag_24", "price_lag_168", "solar_fc", "wind_fc",
                "residual_load_fc", "is_negative"]].copy()
results["price_pred"] = pred
results["neg_prob"] = neg_prob
results.to_parquet(OUT / "predictions_2025.parquet")
importance.rename("importance").to_frame().to_parquet(OUT / "feature_importance.parquet")
joblib.dump({"features": FEATURES, "price_model": reg, "neg_model": clf}, MODELS / "models.joblib")
(OUT / "metrics.json").write_text(json.dumps(
    {"mae": mae, "improvement_vs_naive": improvement, "negative_classifier": cls_metrics}, indent=2))
print("Saved predictions, feature importance, metrics and models.")
