# ============================================================
# NIA26AI021 - WEEK 6 - VID 1
# Advanced ML Models: XGBoost
# Loan Approval Classification
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING LOAN APPROVAL DATASET")
print("=" * 60)

df = pd.read_csv("loan_approval_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. CLEAN DATA
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove leading/trailing spaces from text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Remove duplicate rows
duplicates = df.duplicated().sum()
print("Duplicate rows found:", duplicates)

df = df.drop_duplicates()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. PREPARE TARGET VARIABLE
# ============================================================

# Convert loan status to binary:
# Approved = 1
# Rejected = 0

df["loan_status"] = (
    df["loan_status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "approved": 1,
        "rejected": 0
    })
)

# Check target values
print("\nTarget distribution:")
print(df["loan_status"].value_counts())

# Remove rows where target could not be converted
df = df.dropna(subset=["loan_status"])

df["loan_status"] = df["loan_status"].astype(int)


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("PREPARING FEATURES")
print("=" * 60)

# Drop target and loan ID if present
drop_columns = ["loan_status"]

if "loan_id" in df.columns:
    drop_columns.append("loan_id")

X = df.drop(columns=drop_columns)
y = df["loan_status"]

# Convert categorical columns to numerical dummy variables
X = pd.get_dummies(X, drop_first=True)

# Make sure all values are numeric
X = X.astype(float)

print("Number of features:", X.shape[1])
print("\nFeatures used:")
print(X.columns.tolist())


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:", len(X_train))
print("Testing set:", len(X_test))


# ============================================================
# 6. CALCULATE SCALE_POS_WEIGHT
# ============================================================

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)

negative_samples = (y_train == 0).sum()
positive_samples = (y_train == 1).sum()

scale_pos_weight = negative_samples / positive_samples

print("Negative samples:", negative_samples)
print("Positive samples:", positive_samples)
print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# 7. SCALE FEATURES FOR LOGISTIC REGRESSION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)
logistic_probabilities = logistic_model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]


# ============================================================
# 10. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)
xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, predictions, probabilities):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Rejected", "Approved"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# 12. EVALUATE ALL THREE MODELS
# ============================================================

results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_predictions,
        logistic_probabilities
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions,
        rf_probabilities
    )
)

results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_predictions,
        xgb_probabilities
    )
)


# ============================================================
# 13. FINAL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame(results)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Save comparison table
comparison_df.to_csv(
    "NIA26AI021_Week6_Vid1_Model_Comparison.csv",
    index=False
)

print(
    "\nComparison table saved as "
    "'NIA26AI021_Week6_Vid1_Model_Comparison.csv'"
)


# ============================================================
# 14. XGBOOST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 important features:")
print(feature_importance.head(10).to_string(index=False))


# ============================================================
# 15. FEATURE IMPORTANCE CHART
# ============================================================

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance - Loan Approval")

plt.tight_layout()

plt.savefig(
    "NIA26AI021_Week6_Vid1_XGBoost_Feature_Importance.png",
    dpi=300
)

plt.show()

print(
    "\nFeature importance chart saved as "
    "'NIA26AI021_Week6_Vid1_XGBoost_Feature_Importance.png'"
)


# ============================================================
# 16. IDENTIFY BEST MODELS
# ============================================================

best_f1_model = comparison_df.loc[
    comparison_df["F1 Score"].idxmax()
]

best_auc_model = comparison_df.loc[
    comparison_df["ROC-AUC"].idxmax()
]

best_accuracy_model = comparison_df.loc[
    comparison_df["Accuracy"].idxmax()
]

print("\n" + "=" * 60)
print("BEST RESULTS")
print("=" * 60)

print(
    f"Best F1 Score: "
    f"{best_f1_model['Model']} "
    f"({best_f1_model['F1 Score']:.4f})"
)

print(
    f"Best ROC-AUC: "
    f"{best_auc_model['Model']} "
    f"({best_auc_model['ROC-AUC']:.4f})"
)

print(
    f"Best Accuracy: "
    f"{best_accuracy_model['Model']} "
    f"({best_accuracy_model['Accuracy']:.4f})"
)


# ============================================================
# 17. DEPLOYMENT RECOMMENDATION
# ============================================================

print("\n" + "=" * 60)
print("DEPLOYMENT CONSIDERATION")
print("=" * 60)

print(
    "For a loan approval system, performance alone should not "
    "determine deployment."
)

print(
    "Because loan decisions affect people, explainability, "
    "fairness, and auditability are also important."
)

print(
    "The final deployment choice should therefore consider "
    "the model's F1 score, ROC-AUC, accuracy, explainability, "
    "and training speed."
)

print("\nVid 1 completed successfully!")