# ============================================================
# NIA26AI021 - WEEK 6
# FRAUD DETECTION - VID 2
# Imbalance Handling and Hyperparameter Tuning
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("FRAUD DETECTION - VID 2")
print("=" * 60)

df = pd.read_csv("credit_card_fraud_10k.csv")

print("\nDataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================

target = "is_fraud"

X = df.drop(
    columns=[target, "transaction_id"]
)

y = df[target]


# ============================================================
# 3. CHECK CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)

class_counts = y.value_counts()

negative_samples = class_counts.get(0, 0)
positive_samples = class_counts.get(1, 0)

print("Legitimate transactions:", negative_samples)
print("Fraud transactions:", positive_samples)

scale_pos_weight = (
    negative_samples / positive_samples
)

print(
    f"scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    return {
        "Method": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# 8. BASELINE XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("BASELINE XGBOOST")
print("=" * 60)

baseline_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

baseline_model.fit(
    X_train,
    y_train
)

baseline_result = evaluate_model(
    "Baseline XGBoost",
    baseline_model,
    X_test,
    y_test
)


# ============================================================
# 9. CLASS WEIGHT / SCALE_POS_WEIGHT
# ============================================================

print("\n" + "=" * 60)
print("CLASS WEIGHT XGBOOST")
print("=" * 60)

class_weight_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

class_weight_model.fit(
    X_train,
    y_train
)

class_weight_result = evaluate_model(
    "XGBoost - Class Weight",
    class_weight_model,
    X_test,
    y_test
)


# ============================================================
# 10. SMOTE
# ============================================================

print("\n" + "=" * 60)
print("SMOTE XGBOOST")
print("=" * 60)

smote_model = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

smote_model.fit(
    X_train,
    y_train
)

smote_result = evaluate_model(
    "XGBoost - SMOTE",
    smote_model,
    X_test,
    y_test
)


# ============================================================
# 11. IMBALANCE COMPARISON
# ============================================================

imbalance_results = pd.DataFrame(
    [
        baseline_result,
        class_weight_result,
        smote_result
    ]
)

print("\n" + "=" * 60)
print("IMBALANCE HANDLING COMPARISON")
print("=" * 60)

print(
    imbalance_results.to_string(
        index=False
    )
)


imbalance_file = (
    "NIA26AI021_Week6_Fraud_Vid2_"
    "Imbalance_Comparison.csv"
)

imbalance_results.to_csv(
    imbalance_file,
    index=False
)

print(
    f"\nComparison saved as:"
    f"\n{imbalance_file}"
)


# ============================================================
# 12. BEST IMBALANCE TECHNIQUE
# ============================================================

best_imbalance = imbalance_results.loc[
    imbalance_results["F1 Score"].idxmax()
]

print("\n" + "=" * 60)
print("BEST IMBALANCE TECHNIQUE")
print("=" * 60)

print(
    "Best method:",
    best_imbalance["Method"]
)

print(
    f"Precision: "
    f"{best_imbalance['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_imbalance['Recall']:.4f}"
)

print(
    f"F1 Score: "
    f"{best_imbalance['F1 Score']:.4f}"
)


# ============================================================
# 13. HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

tuning_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


param_distributions = {
    "model__n_estimators": [
        100,
        200,
        300,
        400
    ],

    "model__max_depth": [
        3,
        4,
        5,
        6,
        8
    ],

    "model__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1
    ],

    "model__subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ]
}


random_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(
    X_train,
    y_train
)


# ============================================================
# 14. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(
    random_search.best_params_
)

print(
    f"\nBest cross-validation F1:"
    f" {random_search.best_score_:.4f}"
)


# ============================================================
# 15. EVALUATE TUNED MODEL
# ============================================================

tuned_model = random_search.best_estimator_

tuned_result = evaluate_model(
    "Tuned XGBoost",
    tuned_model,
    X_test,
    y_test
)


# ============================================================
# 16. BEFORE VS AFTER TUNING
# ============================================================

print("\n" + "=" * 60)
print("BEFORE VS AFTER TUNING")
print("=" * 60)

print(
    f"Baseline Accuracy : "
    f"{baseline_result['Accuracy']:.4f}"
)

print(
    f"Tuned Accuracy    : "
    f"{tuned_result['Accuracy']:.4f}"
)

print(
    f"Baseline Precision: "
    f"{baseline_result['Precision']:.4f}"
)

print(
    f"Tuned Precision   : "
    f"{tuned_result['Precision']:.4f}"
)

print(
    f"Baseline Recall   : "
    f"{baseline_result['Recall']:.4f}"
)

print(
    f"Tuned Recall      : "
    f"{tuned_result['Recall']:.4f}"
)

print(
    f"Baseline F1       : "
    f"{baseline_result['F1 Score']:.4f}"
)

print(
    f"Tuned F1          : "
    f"{tuned_result['F1 Score']:.4f}"
)


# ============================================================
# 17. CALCULATE TUNING IMPROVEMENT
# ============================================================

f1_improvement = (
    tuned_result["F1 Score"]
    - baseline_result["F1 Score"]
)

percentage_improvement = (
    f1_improvement
    / baseline_result["F1 Score"]
) * 100


print("\n" + "=" * 60)
print("TUNING IMPROVEMENT")
print("=" * 60)

print(
    f"F1 improvement: "
    f"{f1_improvement:.4f}"
)

print(
    f"Percentage improvement: "
    f"{percentage_improvement:.2f}%"
)


# ============================================================
# 18. SAVE TUNED MODEL
# ============================================================

model_filename = (
    "NIA26AI021_Week6_Fraud_Best_XGBoost.joblib"
)

joblib.dump(
    tuned_model,
    model_filename
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Best model saved as:"
    f"\n{model_filename}"
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VID 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nThe tuned XGBoost model has been saved "
    "and will be used for Vid 3 deployment."
)