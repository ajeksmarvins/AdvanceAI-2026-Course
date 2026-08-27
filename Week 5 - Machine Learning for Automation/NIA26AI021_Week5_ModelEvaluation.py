import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


# =========================================================
# STEP 1: LOAD DATA
# =========================================================

df = pd.read_csv("customer_churn.csv")

print("Customers:", len(df))

print("Churn rate:")
print(df["Churn"].value_counts(normalize=True))


# =========================================================
# STEP 2: EXPLORE AND CLEAN
# =========================================================

print("\nMissing values:")
print(df.isnull().sum())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
    df["MonthlyCharges"].median()
)

# Remove customer ID
df = df.drop(["customerID"], axis=1)


# =========================================================
# STEP 3: PREPARE FEATURES
# =========================================================

le = LabelEncoder()

categorical_cols = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_cols:
    if col != "Churn":
        df[col] = le.fit_transform(df[col])

# Convert target
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop("Churn", axis=1)
y = df["Churn"]


# =========================================================
# STEP 4: TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# STEP 5: SCALE NUMERICAL FEATURES
# =========================================================

scaler = StandardScaler()

numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X_train[numeric_cols] = scaler.fit_transform(
    X_train[numeric_cols]
)

X_test[numeric_cols] = scaler.transform(
    X_test[numeric_cols]
)


# =========================================================
# STEP 6: TRAIN LOGISTIC REGRESSION
# =========================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully.")


# =========================================================
# STEP 7: PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

# Probability of churn
y_prob = model.predict_proba(X_test)[:, 1]


# =========================================================
# STEP 8: ALL MODEL METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")
print(f"ROC-AUC:   {roc_auc:.2%}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stays", "Churns"]
    )
)


# =========================================================
# STEP 9: CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# STEP 10: CONFUSION MATRIX HEATMAP
# =========================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stays", "Churns"],
    yticklabels=["Stays", "Churns"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# =========================================================
# STEP 11: ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# STEP 12: THRESHOLD ANALYSIS
# =========================================================

print("\n========================================")
print("THRESHOLD ANALYSIS")
print("========================================")

# At least five different thresholds
thresholds_to_test = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

threshold_results = []


for threshold in thresholds_to_test:

    threshold_pred = (
        y_prob >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_cm = confusion_matrix(
        y_test,
        threshold_pred
    )

    tn, fp, fn, tp = threshold_cm.ravel()

    # Business calculation
    retention_cost = fp * 50
    retained_value = tp * 500
    net_impact = retained_value - retention_cost

    threshold_results.append({
        "Threshold": threshold,
        "Precision": threshold_precision,
        "Recall": threshold_recall,
        "F1": threshold_f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Retention Cost": retention_cost,
        "Retained Value": retained_value,
        "Net Impact": net_impact
    })


# Convert results to DataFrame
threshold_df = pd.DataFrame(
    threshold_results
)


print("\nThreshold Results:")
print(threshold_df.to_string(index=False))


# =========================================================
# STEP 13: PRECISION VS RECALL PLOT
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall at Different Thresholds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# STEP 14: BUSINESS IMPACT ANALYSIS
# =========================================================

print("\n========================================")
print("BUSINESS IMPACT ANALYSIS")
print("========================================")

print(
    "Retention offer cost per false alarm: $50"
)

print(
    "Value of saving one churning customer: $500"
)

print(
    "\nNet Business Impact = "
    "(Caught Churners × $500) - "
    "(False Alarms × $50)"
)


# Find threshold with highest net impact
best_result = threshold_df.loc[
    threshold_df["Net Impact"].idxmax()
]

best_threshold = best_result["Threshold"]
best_precision = best_result["Precision"]
best_recall = best_result["Recall"]
best_tp = int(best_result["TP"])
best_fp = int(best_result["FP"])
best_fn = int(best_result["FN"])
best_cost = best_result["Retention Cost"]
best_value = best_result["Retained Value"]
best_net = best_result["Net Impact"]


print("\nBest threshold:")
print(f"Threshold: {best_threshold:.2f}")

print(
    f"Precision: {best_precision:.2%}"
)

print(
    f"Recall: {best_recall:.2%}"
)

print(
    f"Caught churners: {best_tp}"
)

print(
    f"False alarms: {best_fp}"
)

print(
    f"Missed churners: {best_fn}"
)

print(
    f"Cost of retention offers: ${best_cost:.2f}"
)

print(
    f"Value of retained customers: ${best_value:.2f}"
)

print(
    f"NET BUSINESS IMPACT: ${best_net:.2f}"
)


# =========================================================
# STEP 15: BUSINESS JUSTIFICATION
# =========================================================

print("\n========================================")
print("BUSINESS JUSTIFICATION")
print("========================================")

print(
    f"Based on the five tested thresholds, "
    f"a threshold of {best_threshold:.2f} "
    f"produced the highest estimated net "
    f"business impact."
)

print(
    f"At this threshold, the model catches "
    f"{best_tp} customers who are likely to churn "
    f"and generates {best_fp} false alarms."
)

print(
    f"The retention offers cost "
    f"${best_cost:.2f}, while the estimated "
    f"value of retained customers is "
    f"${best_value:.2f}."
)

print(
    f"This gives an estimated net business "
    f"impact of ${best_net:.2f}."
)

print(
    "Therefore, this threshold would be "
    "preferred because it gives the highest "
    "business value among the tested thresholds."
)


# =========================================================
# STEP 16: BONUS - RANDOM FOREST
# =========================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

random_forest.fit(
    X_train,
    y_train
)

rf_pred = random_forest.predict(
    X_test
)

rf_prob = random_forest.predict_proba(
    X_test
)[:, 1]


# Random Forest metrics
rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)


print("\n========================================")
print("RANDOM FOREST RESULTS")
print("========================================")

print(f"Accuracy:  {rf_accuracy:.2%}")
print(f"Precision: {rf_precision:.2%}")
print(f"Recall:    {rf_recall:.2%}")
print(f"F1 Score:  {rf_f1:.2%}")
print(f"ROC-AUC:   {rf_auc:.2%}")


# =========================================================
# STEP 17: MODEL COMPARISON
# =========================================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    f"Logistic Regression Accuracy: "
    f"{accuracy:.2%}"
)

print(
    f"Random Forest Accuracy:       "
    f"{rf_accuracy:.2%}"
)

print(
    f"\nLogistic Regression Precision: "
    f"{precision:.2%}"
)

print(
    f"Random Forest Precision:       "
    f"{rf_precision:.2%}"
)

print(
    f"\nLogistic Regression Recall: "
    f"{recall:.2%}"
)

print(
    f"Random Forest Recall:       "
    f"{rf_recall:.2%}"
)

print(
    f"\nLogistic Regression F1: "
    f"{f1:.2%}"
)

print(
    f"Random Forest F1:       "
    f"{rf_f1:.2%}"
)

print(
    f"\nLogistic Regression ROC-AUC: "
    f"{roc_auc:.2%}"
)

print(
    f"Random Forest ROC-AUC:       "
    f"{rf_auc:.2%}"
)


# =========================================================
# STEP 18: FINAL DEPLOYMENT DECISION
# =========================================================

print("\n========================================")
print("FINAL DECISION")
print("========================================")

if rf_recall > recall:

    print(
        "Random Forest wins on recall."
    )

else:

    print(
        "Logistic Regression wins on recall."
    )


if rf_auc > roc_auc:

    print(
        "Random Forest also has the higher ROC-AUC."
    )

else:

    print(
        "Logistic Regression has the higher ROC-AUC."
    )


if rf_recall > recall:

    print(
        "For a churn problem, Random Forest "
        "would be considered for deployment "
        "because recall is important for "
        "catching customers who may churn."
    )

else:

    print(
        "Logistic Regression would be considered "
        "for deployment because it has equal or "
        "better recall."
    )


print("\n=== PRACTICE 2 COMPLETE ===")