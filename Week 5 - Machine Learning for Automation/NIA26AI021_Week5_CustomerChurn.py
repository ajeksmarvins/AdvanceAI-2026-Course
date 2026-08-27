import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv("customer_churn.csv")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# =========================================
# 2. CLEAN THE DATA
# =========================================

# Remove duplicate rows
df = df.drop_duplicates()

# Convert TotalCharges to numeric
# Invalid/blank values become NaN
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove customer ID because it is only an identifier
df = df.drop(
    columns=["customerID"],
    errors="ignore"
)


# =========================================
# 3. CHECK CLEANED DATA
# =========================================

print("\n=== AFTER CLEANING ===")

print("Dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)


# =========================================
# 4. PREPARE FEATURES AND TARGET
# =========================================

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\n=== FEATURES AND TARGET ===")

print("Number of features:", X.shape[1])

print("Target distribution:")
print(y.value_counts())

print("\nTarget proportions:")
print(y.value_counts(normalize=True))


# =========================================
# 5. TRAIN / TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== TRAIN / TEST SPLIT ===")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================================
# 6. IDENTIFY FEATURES
# =========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# =========================================
# 7. PREPROCESSING PIPELINE
# =========================================

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
            OneHotEncoder(handle_unknown="ignore")
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

print("\nPreprocessing pipeline created successfully.")


# =========================================
# 8. LOGISTIC REGRESSION MODEL
# =========================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

print(
    "\nLogistic Regression model "
    "trained successfully."
)


# =========================================
# 9. EVALUATE LOGISTIC REGRESSION
# =========================================

y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

logistic_precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n=== LOGISTIC REGRESSION RESULTS ===")

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {logistic_precision:.2%}")
print(f"Recall: {logistic_recall:.2%}")
print(f"F1 Score: {logistic_f1:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(cm)


# =========================================
# 10. BUSINESS IMPACT ANALYSIS
# =========================================

tn, fp, fn, tp = cm.ravel()

false_alarms = fp
caught_churn = tp
missed_churn = fn

print("\n=== BUSINESS IMPACT ===")

print(f"Caught churn customers: {caught_churn}")
print(f"Missed churn customers: {missed_churn}")
print(f"False alarms: {false_alarms}")

# Business assumptions from the practice example
cost_of_retention = false_alarms * 50
value_of_retained = caught_churn * 500 * 0.5

net_impact = value_of_retained - cost_of_retention

print(f"\nCost of retention offers: ${cost_of_retention}")
print(f"Value of retained customers: ${value_of_retained}")
print(f"NET IMPACT: ${net_impact}")

# =========================================
# 11. DEPLOYMENT DECISION
# =========================================

print("\n=== DEPLOYMENT DECISION ===")

if net_impact > 0:
    print("Decision: The model may be worth deploying.")
    print(f"The estimated net business impact is ${net_impact}.")
    print(
        f"The model caught {caught_churn} churners "
        f"while generating {false_alarms} false alarms."
    )
else:
    print("Decision: Do not deploy the model yet.")
    print(f"The estimated net business impact is ${net_impact}.")
    
# =========================================
# 12. BONUS: RANDOM FOREST
# =========================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

print(
    "\nRandom Forest model "
    "trained successfully."
)


# =========================================
# 13. RANDOM FOREST EVALUATION
# =========================================

rf_pred = random_forest_model.predict(
    X_test
)

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

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)

print("\n=== RANDOM FOREST RESULTS ===")

print(f"Accuracy: {rf_accuracy:.2%}")
print(f"Precision: {rf_precision:.2%}")
print(f"Recall: {rf_recall:.2%}")
print(f"F1 Score: {rf_f1:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred
    )
)

print("\nConfusion Matrix:")
print(rf_cm)


# =========================================
# 14. MODEL COMPARISON
# =========================================

print(
    "\n=== LOGISTIC REGRESSION "
    "vs RANDOM FOREST ==="
)

print(
    f"Logistic Regression Accuracy: "
    f"{accuracy:.2%}"
)

print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy:.2%}"
)

print(
    f"\nLogistic Regression Recall: "
    f"{logistic_recall:.2%}"
)

print(
    f"Random Forest Recall: "
    f"{rf_recall:.2%}"
)

print(
    f"\nLogistic Regression F1: "
    f"{logistic_f1:.2%}"
)

print(
    f"Random Forest F1: "
    f"{rf_f1:.2%}"
)


# =========================================
# 15. BONUS CONCLUSION
# =========================================

if rf_recall > logistic_recall:
    print(
        "\nRandom Forest has better recall."
    )
else:
    print(
        "\nLogistic Regression has better "
        "or equal recall."
    )


if rf_accuracy > accuracy:
    print(
        "Random Forest has better accuracy."
    )
else:
    print(
        "Logistic Regression has better "
        "or equal accuracy."
    )


print("\n=== PRACTICE 1 COMPLETE ===")