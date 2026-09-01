import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# WEEK 8 VID 1
# CROSS-VALIDATION AND HYPERPARAMETER TUNING
# ============================================================

print("=" * 70)
print("WEEK 8 VID 1 - CROSS-VALIDATION & HYPERPARAMETER TUNING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("credit_card_fraud_10k.csv")

print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

TARGET = "is_fraud"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )


# Use numerical columns only
numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()


# Remove target from the feature list
feature_columns = [
    col
    for col in numeric_columns
    if col != TARGET
]


# Remove obvious identifier columns
identifier_keywords = [
    "id",
    "transaction_id"
]

feature_columns = [
    col
    for col in feature_columns
    if not any(
        keyword in col.lower()
        for keyword in identifier_keywords
    )
]


if len(feature_columns) == 0:
    raise ValueError(
        "No usable numerical feature columns were found."
    )


X = df[feature_columns].copy()
y = df[TARGET].copy()


# Handle missing numerical values
X = X.fillna(X.median())


print("\nFeatures used:")
for feature in feature_columns:
    print(f"- {feature}")


print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. CHECK CLASS IMBALANCE
# ============================================================

negative_count = (y == 0).sum()
positive_count = (y == 1).sum()

if positive_count == 0:
    raise ValueError(
        "No positive fraud records were found."
    )

scale_pos_weight = (
    negative_count / positive_count
)

print("\n" + "=" * 70)
print("CLASS IMBALANCE")
print("=" * 70)

print(f"Normal transactions: {negative_count}")
print(f"Fraud transactions : {positive_count}")
print(
    f"scale_pos_weight   : "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ============================================================
# 5. DEFAULT XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("DEFAULT XGBOOST MODEL")
print("=" * 70)

default_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


default_model.fit(
    X_train,
    y_train
)


# Single split predictions
default_predictions = default_model.predict(
    X_test
)


# ============================================================
# 6. SINGLE SPLIT METRICS
# ============================================================

single_precision = precision_score(
    y_test,
    default_predictions,
    zero_division=0
)

single_recall = recall_score(
    y_test,
    default_predictions,
    zero_division=0
)

single_f1 = f1_score(
    y_test,
    default_predictions,
    zero_division=0
)

single_accuracy = accuracy_score(
    y_test,
    default_predictions
)


print("\nSingle train/test split results:")
print(
    f"Accuracy : {single_accuracy:.4f}"
)
print(
    f"Precision: {single_precision:.4f}"
)
print(
    f"Recall   : {single_recall:.4f}"
)
print(
    f"F1 Score : {single_f1:.4f}"
)


print("\nClassification report:")
print(
    classification_report(
        y_test,
        default_predictions,
        zero_division=0
    )
)


print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        default_predictions
    )
)


# ============================================================
# 7. FIVE-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_results = cross_validate(
    default_model,
    X,
    y,
    cv=cv,
    scoring={
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    },
    n_jobs=-1
)


precision_scores = cv_results[
    "test_precision"
]

recall_scores = cv_results[
    "test_recall"
]

f1_scores = cv_results[
    "test_f1"
]


precision_mean = precision_scores.mean()
precision_std = precision_scores.std()

recall_mean = recall_scores.mean()
recall_std = recall_scores.std()

f1_mean = f1_scores.mean()
f1_std = f1_scores.std()


print("\nFold-by-fold results:")

for i in range(5):

    print(
        f"Fold {i + 1}: "
        f"Precision={precision_scores[i]:.4f}, "
        f"Recall={recall_scores[i]:.4f}, "
        f"F1={f1_scores[i]:.4f}"
    )


print("\nCross-validation summary:")

print(
    f"Precision: "
    f"{precision_mean:.4f} "
    f"+/- {precision_std:.4f}"
)

print(
    f"Recall   : "
    f"{recall_mean:.4f} "
    f"+/- {recall_std:.4f}"
)

print(
    f"F1 Score : "
    f"{f1_mean:.4f} "
    f"+/- {f1_std:.4f}"
)


# ============================================================
# 8. COMPARE SINGLE SPLIT VS CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SINGLE SPLIT VS CROSS-VALIDATION")
print("=" * 70)

comparison_df = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1"
    ],

    "Single Split": [
        single_precision,
        single_recall,
        single_f1
    ],

    "CV Mean": [
        precision_mean,
        recall_mean,
        f1_mean
    ],

    "CV Std": [
        precision_std,
        recall_std,
        f1_std
    ]
})


print(
    comparison_df.round(4).to_string(
        index=False
    )
)


comparison_df.to_csv(
    "cross_validation_comparison.csv",
    index=False
)

print(
    "\nComparison saved as "
    "'cross_validation_comparison.csv'"
)


# ============================================================
# 9. GRIDSEARCHCV
# ============================================================

print("\n" + "=" * 70)
print("GRIDSEARCHCV")
print("=" * 70)

print(
    "\nSearching for the best XGBoost parameters "
    "using 5-fold cross-validation..."
)


grid_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


param_grid = {
    "n_estimators": [
        100,
        200
    ],

    "max_depth": [
        3,
        5,
        7
    ],

    "learning_rate": [
        0.05,
        0.1
    ],

    "subsample": [
        0.8,
        1.0
    ]
}


grid_search = GridSearchCV(
    estimator=grid_model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


print("\nBest parameters:")

for parameter, value in (
    grid_search.best_params_.items()
):

    print(
        f"{parameter}: {value}"
    )


print(
    f"\nBest cross-validation F1: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 10. TUNED MODEL EVALUATION
# ============================================================

tuned_model = grid_search.best_estimator_


tuned_predictions = tuned_model.predict(
    X_test
)


tuned_precision = precision_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)


print("\n" + "=" * 70)
print("TUNED XGBOOST RESULTS")
print("=" * 70)

print(
    f"Accuracy : {tuned_accuracy:.4f}"
)

print(
    f"Precision: {tuned_precision:.4f}"
)

print(
    f"Recall   : {tuned_recall:.4f}"
)

print(
    f"F1 Score : {tuned_f1:.4f}"
)


print("\nClassification report:")
print(
    classification_report(
        y_test,
        tuned_predictions,
        zero_division=0
    )
)


# ============================================================
# 11. DEFAULT VS TUNED PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("DEFAULT VS TUNED MODEL")
print("=" * 70)


f1_improvement = (
    tuned_f1 - single_f1
)

if single_f1 != 0:

    f1_improvement_percent = (
        f1_improvement
        / single_f1
        * 100
    )

else:

    f1_improvement_percent = 0


performance_comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ],

    "Default": [
        single_accuracy,
        single_precision,
        single_recall,
        single_f1
    ],

    "Tuned": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1
    ]
})


performance_comparison[
    "Improvement"
] = (
    performance_comparison["Tuned"]
    - performance_comparison["Default"]
)


print(
    performance_comparison.round(4)
    .to_string(index=False)
)


print(
    f"\nF1 improvement: "
    f"{f1_improvement:.4f}"
)

print(
    f"F1 improvement percentage: "
    f"{f1_improvement_percent:.2f}%"
)


performance_comparison.to_csv(
    "default_vs_tuned.csv",
    index=False
)

print(
    "\nPerformance comparison saved as "
    "'default_vs_tuned.csv'"
)


# ============================================================
# 12. CROSS-VALIDATION OF TUNED MODEL
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION OF TUNED MODEL")
print("=" * 70)


tuned_cv_results = cross_validate(
    tuned_model,
    X,
    y,
    cv=cv,
    scoring={
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    },
    n_jobs=-1
)


tuned_precision_scores = (
    tuned_cv_results["test_precision"]
)

tuned_recall_scores = (
    tuned_cv_results["test_recall"]
)

tuned_f1_scores = (
    tuned_cv_results["test_f1"]
)


tuned_precision_mean = (
    tuned_precision_scores.mean()
)

tuned_precision_std = (
    tuned_precision_scores.std()
)

tuned_recall_mean = (
    tuned_recall_scores.mean()
)

tuned_recall_std = (
    tuned_recall_scores.std()
)

tuned_f1_mean = (
    tuned_f1_scores.mean()
)

tuned_f1_std = (
    tuned_f1_scores.std()
)


print(
    f"Precision: "
    f"{tuned_precision_mean:.4f} "
    f"+/- {tuned_precision_std:.4f}"
)

print(
    f"Recall   : "
    f"{tuned_recall_mean:.4f} "
    f"+/- {tuned_recall_std:.4f}"
)

print(
    f"F1 Score : "
    f"{tuned_f1_mean:.4f} "
    f"+/- {tuned_f1_std:.4f}"
)


# ============================================================
# 13. DEPLOYMENT RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("DEPLOYMENT RECOMMENDATION")
print("=" * 70)


if (
    tuned_f1_mean >= 0.80
    and tuned_f1_std <= 0.10
):

    recommendation = """
I would consider deploying the tuned XGBoost model.

The model has strong average cross-validation performance
and relatively low variation between folds, suggesting that
its performance is reasonably stable.

Before production deployment, I would still validate the
model on fresh unseen data and monitor fraud detection
performance over time.
"""

else:

    recommendation = """
I would not deploy the model immediately.

The cross-validation results suggest that more validation,
data improvement, or further tuning may be needed before
production deployment.

A reliable production model should have strong average
performance as well as acceptable variation across folds.
"""


print(
    recommendation
)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

import joblib


joblib.dump(
    tuned_model,
    "best_xgboost_week8.joblib"
)


joblib.dump(
    feature_columns,
    "feature_columns_week8.joblib"
)


print(
    "\nTuned model saved as "
    "'best_xgboost_week8.joblib'"
)

print(
    "Feature columns saved as "
    "'feature_columns_week8.joblib'"
)


# ============================================================
# 15. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Default single-split F1: "
    f"{single_f1:.4f}"
)

print(
    f"Default CV F1: "
    f"{f1_mean:.4f} "
    f"+/- {f1_std:.4f}"
)

print(
    f"Tuned test F1: "
    f"{tuned_f1:.4f}"
)

print(
    f"Tuned CV F1: "
    f"{tuned_f1_mean:.4f} "
    f"+/- {tuned_f1_std:.4f}"
)

print(
    f"F1 improvement: "
    f"{f1_improvement_percent:.2f}%"
)


print("\n" + "=" * 70)
print("VID 1 COMPLETED SUCCESSFULLY!")
print("=" * 70)