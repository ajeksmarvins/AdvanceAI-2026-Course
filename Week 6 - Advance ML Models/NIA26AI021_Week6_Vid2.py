# ============================================================
# NIA26AI021 - WEEK 6 - VID 2
# Imbalance Handling + Hyperparameter Tuning
# Loan Approval Classification
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING LOAN APPROVAL DATASET")
print("=" * 60)

df = pd.read_csv("loan_approval_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Clean text values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()


# ============================================================
# 2. PREPARE TARGET
# ============================================================

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

df = df.dropna(subset=["loan_status"])

df["loan_status"] = df["loan_status"].astype(int)


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

drop_columns = ["loan_status"]

if "loan_id" in df.columns:
    drop_columns.append("loan_id")

X = df.drop(columns=drop_columns)
y = df["loan_status"]

# Convert categorical variables to numerical variables
X = pd.get_dummies(X, drop_first=True)

X = X.astype(float)


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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(y_train.value_counts())

negative_samples = (y_train == 0).sum()
positive_samples = (y_train == 1).sum()

scale_pos_weight = negative_samples / positive_samples

print("\nNegative samples:", negative_samples)
print("Positive samples:", positive_samples)
print("Imbalance ratio:", round(scale_pos_weight, 4))


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X_data, y_data):

    predictions = model.predict(X_data)

    precision = precision_score(
        y_data,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_data,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_data,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_data,
        predictions
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {
        "Method": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# ============================================================
# 7. BASELINE RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("BASELINE RANDOM FOREST")
print("=" * 60)

baseline_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

baseline_model.fit(X_train, y_train)

results = []

results.append(
    evaluate_model(
        "Baseline Random Forest",
        baseline_model,
        X_test,
        y_test
    )
)


# ============================================================
# 8. CLASS WEIGHT TECHNIQUE
# ============================================================

print("\n" + "=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)

class_weight_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

class_weight_model.fit(X_train, y_train)

results.append(
    evaluate_model(
        "Random Forest - Class Weights",
        class_weight_model,
        X_test,
        y_test
    )
)


# ============================================================
# 9. SMOTE
# ============================================================

print("\n" + "=" * 60)
print("SMOTE")
print("=" * 60)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Original training distribution:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


smote_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

smote_model.fit(
    X_train_smote,
    y_train_smote
)

results.append(
    evaluate_model(
        "Random Forest - SMOTE",
        smote_model,
        X_test,
        y_test
    )
)


# ============================================================
# 10. IMBALANCE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("IMBALANCE HANDLING COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame(results)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

comparison_df.to_csv(
    "NIA26AI021_Week6_Vid2_Imbalance_Comparison.csv",
    index=False
)

print(
    "\nSaved: NIA26AI021_Week6_Vid2_Imbalance_Comparison.csv"
)


# ============================================================
# 11. FIND BEST RECALL AND BEST BALANCE
# ============================================================

best_recall = comparison_df.loc[
    comparison_df["Recall"].idxmax()
]

best_f1 = comparison_df.loc[
    comparison_df["F1 Score"].idxmax()
]

print("\n" + "=" * 60)
print("IMBALANCE RESULTS")
print("=" * 60)

print(
    f"Best Recall: {best_recall['Method']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    f"Best F1 Score / Balance: {best_f1['Method']} "
    f"({best_f1['F1 Score']:.4f})"
)


# ============================================================
# 12. HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

parameter_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}


random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=parameter_grid,
    n_iter=15,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)


# ============================================================
# 13. BEST PARAMETERS
# ============================================================

print("\nBest parameters:")
print(random_search.best_params_)

print(
    "\nBest cross-validation F1:",
    round(random_search.best_score_, 4)
)


# ============================================================
# 14. EVALUATE TUNED MODEL
# ============================================================

tuned_model = random_search.best_estimator_

tuned_results = evaluate_model(
    "Tuned Random Forest",
    tuned_model,
    X_test,
    y_test
)


# ============================================================
# 15. BEFORE VS AFTER TUNING
# ============================================================

baseline_row = comparison_df[
    comparison_df["Method"] == "Baseline Random Forest"
].iloc[0]

print("\n" + "=" * 60)
print("BEFORE VS AFTER TUNING")
print("=" * 60)

print(
    f"Baseline Accuracy: "
    f"{baseline_row['Accuracy']:.4f}"
)

print(
    f"Tuned Accuracy: "
    f"{tuned_results['Accuracy']:.4f}"
)

print(
    f"Baseline Precision: "
    f"{baseline_row['Precision']:.4f}"
)

print(
    f"Tuned Precision: "
    f"{tuned_results['Precision']:.4f}"
)

print(
    f"Baseline Recall: "
    f"{baseline_row['Recall']:.4f}"
)

print(
    f"Tuned Recall: "
    f"{tuned_results['Recall']:.4f}"
)

print(
    f"Baseline F1: "
    f"{baseline_row['F1 Score']:.4f}"
)

print(
    f"Tuned F1: "
    f"{tuned_results['F1 Score']:.4f}"
)


# ============================================================
# 16. CALCULATE IMPROVEMENT
# ============================================================

f1_improvement = (
    tuned_results["F1 Score"]
    - baseline_row["F1 Score"]
)

f1_improvement_percentage = (
    f1_improvement
    / baseline_row["F1 Score"]
) * 100

print("\n" + "=" * 60)
print("TUNING IMPROVEMENT")
print("=" * 60)

print(
    f"F1 improvement: {f1_improvement:.4f}"
)

print(
    f"Percentage improvement: "
    f"{f1_improvement_percentage:.2f}%"
)


# ============================================================
# 17. SAVE TUNED MODEL
# ============================================================

model_filename = (
    "NIA26AI021_Week6_Vid2_Best_RandomForest.joblib"
)

joblib.dump(
    tuned_model,
    model_filename
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Best model saved as: {model_filename}"
)

print("\nVid 2 completed successfully!")