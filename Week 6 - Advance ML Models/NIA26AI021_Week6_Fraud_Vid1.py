# ============================================================
# NIA26AI021 - WEEK 6
# FRAUD DETECTION - VID 1
# Advanced ML Model Comparison
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("FRAUD DETECTION - VID 1")
print("=" * 60)

df = pd.read_csv("credit_card_fraud_10k.csv")

print("\nDataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. TARGET AND FEATURES
# ============================================================

target = "is_fraud"

# Remove transaction ID because it is an identifier,
# not a useful predictive feature.

drop_columns = ["transaction_id"]

X = df.drop(
    columns=[target] + drop_columns
)

y = df[target]


# ============================================================
# 4. CLASS BALANCE
# ============================================================

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)

class_counts = y.value_counts()

print("\nLegitimate transactions (0):",
      class_counts.get(0, 0))

print("Fraud transactions (1):",
      class_counts.get(1, 0))

negative_samples = class_counts.get(0, 0)
positive_samples = class_counts.get(1, 0)

scale_pos_weight = negative_samples / positive_samples

print(
    f"\nscale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 6. PREPROCESSING
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
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
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
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
}


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

trained_models = {}


for model_name, model in models.items():

    print("\n")
    print("=" * 60)
    print(model_name.upper())
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
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

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

    trained_models[model_name] = pipeline

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


# ============================================================
# 10. FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    results
).T

comparison.index.name = "Model"

comparison = comparison.reset_index()

comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE COMPARISON TABLE
# ============================================================

comparison_file = (
    "NIA26AI021_Week6_Fraud_Vid1_Model_Comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

print(
    f"\nComparison table saved as:"
    f"\n{comparison_file}"
)


# ============================================================
# 12. FIND BEST MODEL
# ============================================================

best_row = comparison.iloc[0]

best_model_name = best_row["Model"]

best_f1 = best_row["F1 Score"]

best_accuracy = best_row["Accuracy"]

best_roc_auc = best_row["ROC-AUC"]


print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Best model by F1 Score: "
    f"{best_model_name}"
)

print(
    f"F1 Score : {best_f1:.4f}"
)

print(
    f"Accuracy : {best_accuracy:.4f}"
)

print(
    f"ROC-AUC  : {best_roc_auc:.4f}"
)


# ============================================================
# 13. DEPLOYMENT RECOMMENDATION
# ============================================================

print("\n")
print("=" * 60)
print("DEPLOYMENT RECOMMENDATION")
print("=" * 60)

print(
    f"""
For this fraud detection system, I would deploy
{best_model_name} because it achieved the highest
F1 score among the models tested.

Fraud detection is an imbalanced classification problem,
so F1 score, precision, recall and ROC-AUC are more useful
than accuracy alone.

The model should also be evaluated for training speed,
explainability and reliability before production deployment.

For an automated fraud system, the final model should
produce a fraud probability that can be used to assign
risk levels and trigger appropriate actions.
"""
)


# ============================================================
# 14. END
# ============================================================

print("\n" + "=" * 60)
print("VID 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)


print("\n" + "=" * 60)
print("TARGET CORRELATION CHECK")
print("=" * 60)

numeric_df = df.select_dtypes(include=["number"])

if "is_fraud" in numeric_df.columns:
    print(
        numeric_df.corr()["is_fraud"]
        .sort_values(ascending=False)
    )
