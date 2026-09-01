# ============================================================
# MONTH 2 PROJECT
# INTELLIGENT DECISION AUTOMATOR
#
# PROJECT: FRAUD DETECTION
# ============================================================

import os
import sqlite3
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.compose import ColumnTransformer
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

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "fraud_transactions.csv"
)

NEW_DATA_PATH = os.path.join(
    "data",
    "new_transactions.csv"
)

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

TARGET = "is_fraud"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Decision thresholds
CONFIDENCE_AUTO_ACTION = 0.80
CONFIDENCE_REVIEW = 0.50

# Illustrative business assumptions
COST_FALSE_NEGATIVE = 100000
COST_FALSE_POSITIVE = 5000
VALUE_TRUE_POSITIVE = 100000


# ============================================================
# 2. OUTPUT PATHS
# ============================================================

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fraud_detection_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "fraud_preprocessor.pkl"
)

FEATURE_SCHEMA_PATH = os.path.join(
    MODEL_DIR,
    "feature_schema.pkl"
)

PREDICTION_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "fraud_predictions.csv"
)

MODEL_COMPARISON_PATH = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

FINAL_METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "final_model_metrics.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

TARGET_DISTRIBUTION_PATH = os.path.join(
    OUTPUT_DIR,
    "target_distribution.png"
)

BUSINESS_IMPACT_PATH = os.path.join(
    OUTPUT_DIR,
    "business_impact.csv"
)

CROSS_VALIDATION_PATH = os.path.join(
    OUTPUT_DIR,
    "cross_validation_results.csv"
)

DATABASE_PATH = os.path.join(
    OUTPUT_DIR,
    "fraud_predictions.db"
)

LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "automation.log"
)


# ============================================================
# 3. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_PATH,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# 5. PROJECT HEADER
# ============================================================

print("=" * 75)
print("MONTH 2 PROJECT - INTELLIGENT DECISION AUTOMATOR")
print("FRAUD DETECTION SYSTEM")
print("=" * 75)


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("1. DATASET LOADING")
print("=" * 75)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Please make sure fraud_transactions.csv is inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded successfully: "
    f"{df.shape[0]:,} rows x {df.shape[1]} columns"
)


# ============================================================
# 7. BASIC DATA CLEANING
# ============================================================

print("\n" + "=" * 75)
print("2. DATA CLEANING")
print("=" * 75)

# Remove accidental spaces from column names
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

# Remove duplicate rows
duplicate_count = int(
    df.duplicated().sum()
)

print(
    f"Duplicate rows found: {duplicate_count:,}"
)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(
        f"Duplicates removed. "
        f"Remaining rows: {len(df):,}"
    )
else:
    print("No duplicate rows found.")


# ============================================================
# 8. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 75)

print("\nFirst five rows:")
print(
    df.head().to_string(index=False)
)

print("\nDataset information:")
print(
    df.info()
)

print("\nMissing values:")
print(
    df.isnull().sum()
)

print("\nDescriptive statistics:")
print(
    df.describe(include="all").transpose()
)


# ============================================================
# 9. TARGET VALIDATION
# ============================================================

if TARGET not in df.columns:
    raise KeyError(
        f"Target column '{TARGET}' was not found in the dataset."
    )

# Convert target to integer where possible
df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

# Remove rows with missing target
missing_target = int(
    df[TARGET].isnull().sum()
)

if missing_target > 0:

    print(
        f"\nRemoving {missing_target} rows "
        f"with missing target values."
    )

    df = df.dropna(
        subset=[TARGET]
    )

df[TARGET] = df[TARGET].astype(int)


# Make sure target contains only 0 and 1
invalid_target_values = sorted(
    set(df[TARGET].unique()) - {0, 1}
)

if invalid_target_values:

    raise ValueError(
        "Target column must contain only 0 and 1. "
        f"Found: {invalid_target_values}"
    )


fraud_count = int(
    (df[TARGET] == 1).sum()
)

legitimate_count = int(
    (df[TARGET] == 0).sum()
)

if fraud_count == 0 or legitimate_count == 0:

    raise ValueError(
        "The dataset must contain both legitimate (0) "
        "and fraudulent (1) transactions."
    )

fraud_rate = (
    fraud_count / len(df)
) * 100

imbalance_ratio = (
    legitimate_count / fraud_count
)


print("\nTarget distribution:")
print(
    df[TARGET].value_counts()
)

print("\nTarget percentage:")
print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(
    f"\nLegitimate transactions: "
    f"{legitimate_count:,}"
)

print(
    f"Fraudulent transactions: "
    f"{fraud_count:,}"
)

print(
    f"Fraud rate: "
    f"{fraud_rate:.2f}%"
)

print(
    f"Legitimate-to-fraud ratio: "
    f"{imbalance_ratio:.2f}:1"
)


# ============================================================
# 10. TARGET VISUALIZATION
# ============================================================

plt.figure(
    figsize=(7, 5)
)

df[TARGET].value_counts().sort_index().plot(
    kind="bar"
)

plt.title(
    "Fraud vs Legitimate Transactions"
)

plt.xlabel(
    "Transaction Class "
    "(0 = Legitimate, 1 = Fraud)"
)

plt.ylabel(
    "Number of Transactions"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    TARGET_DISTRIBUTION_PATH,
    dpi=300
)

plt.close()


# ============================================================
# 11. DEFINE FEATURES
# ============================================================

feature_columns = [
    "transaction_amount",
    "account_age_days",
    "transaction_hour",
    "previous_transactions",
    "device_type",
    "location_match",
    "amount_vs_average",
    "is_new_device"
]


missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:

    raise KeyError(
        "The dataset is missing these required features:\n"
        f"{missing_features}"
    )


X = df[
    feature_columns
].copy()

y = df[
    TARGET
].copy()


# ============================================================
# 12. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = [
    "transaction_amount",
    "account_age_days",
    "transaction_hour",
    "previous_transactions",
    "amount_vs_average"
]

categorical_features = [
    "device_type",
    "location_match",
    "is_new_device"
]


print("\nNumeric features:")

for feature in numeric_features:
    print(
        f"- {feature}"
    )


print("\nCategorical features:")

for feature in categorical_features:
    print(
        f"- {feature}"
    )


# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 75)
print("4. TRAIN / TEST SPLIT")
print("=" * 75)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"Training records: "
    f"{len(X_train):,}"
)

print(
    f"Testing records: "
    f"{len(X_test):,}"
)


# ============================================================
# 14. PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 75)
print("5. PREPROCESSING")
print("=" * 75)

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([
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
])


print(
    "Missing numeric values → median imputation"
)

print(
    "Numeric features → StandardScaler"
)

print(
    "Missing categorical values → most-frequent imputation"
)

print(
    "Categorical features → OneHotEncoder"
)

print(
    "Preprocessing is performed inside each ML pipeline "
    "to reduce data leakage."
)


# ============================================================
# 15. CLASS IMBALANCE
# ============================================================

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

if positive_count == 0:

    raise ValueError(
        "Training data contains no fraudulent transactions."
    )

scale_pos_weight = (
    negative_count / positive_count
)

print("\nClass imbalance handling:")

print(
    f"Legitimate training records: "
    f"{negative_count:,}"
)

print(
    f"Fraudulent training records: "
    f"{positive_count:,}"
)

print(
    f"XGBoost scale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# 16. DEFINE MACHINE LEARNING MODELS
# ============================================================

lr_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])


rf_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


xgb_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}


# ============================================================
# 17. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_data,
    y_data
):

    predictions = model.predict(
        X_data
    )

    probabilities = model.predict_proba(
        X_data
    )[:, 1]

    return {
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_data,
            predictions
        ),

        "Precision": precision_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_data,
            probabilities
        )
    }


# ============================================================
# 18. TRAIN AND COMPARE MODELS
# ============================================================

print("\n" + "=" * 75)
print("6. MODEL TRAINING AND COMPARISON")
print("=" * 75)

model_results = []

trained_models = {}


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    trained_models[
        model_name
    ] = model

    result = evaluate_model(
        model_name,
        model,
        X_test,
        y_test
    )

    model_results.append(
        result
    )

    print(
        f"Accuracy : {result['Accuracy']:.3f}"
    )

    print(
        f"Precision: {result['Precision']:.3f}"
    )

    print(
        f"Recall   : {result['Recall']:.3f}"
    )

    print(
        f"F1 Score : {result['F1 Score']:.3f}"
    )

    print(
        f"ROC-AUC  : {result['ROC-AUC']:.3f}"
    )


comparison = pd.DataFrame(
    model_results
)


# For fraud detection, F1 is prioritized because
# we need a balance between detecting fraud and
# avoiding excessive false alerts.

comparison = comparison.sort_values(
    by=[
        "F1 Score",
        "Recall",
        "ROC-AUC"
    ],
    ascending=False
).reset_index(
    drop=True
)


print("\nMODEL COMPARISON")

print(
    comparison.to_string(
        index=False
    )
)


comparison.to_csv(
    MODEL_COMPARISON_PATH,
    index=False
)


# ============================================================
# 19. STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("7. STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 75)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


cv_results = []


for model_name, model in models.items():

    print(
        f"\nCross-validating {model_name}..."
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    cv_results.append({

        "Model": model_name,

        "CV Precision Mean":
            scores[
                "test_precision"
            ].mean(),

        "CV Precision Std":
            scores[
                "test_precision"
            ].std(),

        "CV Recall Mean":
            scores[
                "test_recall"
            ].mean(),

        "CV Recall Std":
            scores[
                "test_recall"
            ].std(),

        "CV F1 Mean":
            scores[
                "test_f1"
            ].mean(),

        "CV F1 Std":
            scores[
                "test_f1"
            ].std(),

        "CV ROC-AUC Mean":
            scores[
                "test_roc_auc"
            ].mean(),

        "CV ROC-AUC Std":
            scores[
                "test_roc_auc"
            ].std()
    })


cv_df = pd.DataFrame(
    cv_results
)


cv_df = cv_df.sort_values(
    by="CV F1 Mean",
    ascending=False
).reset_index(
    drop=True
)


print("\nCROSS-VALIDATION RESULTS")

print(
    cv_df.to_string(
        index=False
    )
)


cv_df.to_csv(
    CROSS_VALIDATION_PATH,
    index=False
)


# ============================================================
# 20. SELECT MODEL FOR HYPERPARAMETER TUNING
# ============================================================

best_base_model_name = (
    cv_df.iloc[0]["Model"]
)

print(
    "\nModel selected for hyperparameter tuning:"
)

print(
    best_base_model_name
)


base_model = models[
    best_base_model_name
]


# ============================================================
# 21. HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 75)
print("8. HYPERPARAMETER TUNING")
print("=" * 75)


if best_base_model_name == "Logistic Regression":

    param_grid = {

        "classifier__C": [
            0.01,
            0.1,
            1,
            10
        ],

        "classifier__solver": [
            "liblinear",
            "lbfgs"
        ]
    }


elif best_base_model_name == "Random Forest":

    param_grid = {

        "classifier__n_estimators": [
            100,
            200,
            300
        ],

        "classifier__max_depth": [
            None,
            8,
            12,
            20
        ],

        "classifier__min_samples_split": [
            2,
            5
        ],

        "classifier__min_samples_leaf": [
            1,
            2
        ]
    }


else:

    param_grid = {

        "classifier__n_estimators": [
            100,
            200,
            300
        ],

        "classifier__max_depth": [
            3,
            5,
            7
        ],

        "classifier__learning_rate": [
            0.03,
            0.05,
            0.1
        ],

        "classifier__subsample": [
            0.8,
            1.0
        ]
    }


grid_search = GridSearchCV(
    estimator=base_model,
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


best_model = (
    grid_search.best_estimator_
)


print(
    "\nBest parameters:"
)

print(
    grid_search.best_params_
)


print(
    "\nBest cross-validation F1:"
)

print(
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 22. FINAL MODEL EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("9. FINAL MODEL EVALUATION")
print("=" * 75)


final_predictions = best_model.predict(
    X_test
)

final_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_roc_auc = roc_auc_score(
    y_test,
    final_probabilities
)


print(
    f"Final model: "
    f"{best_base_model_name}"
)

print(
    f"Accuracy : "
    f"{final_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{final_precision:.4f}"
)

print(
    f"Recall   : "
    f"{final_recall:.4f}"
)

print(
    f"F1 Score : "
    f"{final_f1:.4f}"
)

print(
    f"ROC-AUC  : "
    f"{final_roc_auc:.4f}"
)


# ============================================================
# 23. CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("10. CONFUSION MATRIX")
print("=" * 75)


cm = confusion_matrix(
    y_test,
    final_predictions
)


print(cm)


plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Legitimate",
        "Fraud"
    ],
    yticklabels=[
        "Legitimate",
        "Fraud"
    ]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Fraud Detection Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)

plt.close()


# ============================================================
# 25. BUSINESS IMPACT
# ============================================================

print("\n" + "=" * 75)
print("11. BUSINESS IMPACT")
print("=" * 75)


true_negative = int(
    cm[0, 0]
)

false_positive = int(
    cm[0, 1]
)

false_negative = int(
    cm[1, 0]
)

true_positive = int(
    cm[1, 1]
)


print(
    f"True Negatives : "
    f"{true_negative}"
)

print(
    f"False Positives: "
    f"{false_positive}"
)

print(
    f"False Negatives: "
    f"{false_negative}"
)

print(
    f"True Positives : "
    f"{true_positive}"
)


false_negative_cost = (
    false_negative
    * COST_FALSE_NEGATIVE
)

false_positive_cost = (
    false_positive
    * COST_FALSE_POSITIVE
)

prevented_fraud_value = (
    true_positive
    * VALUE_TRUE_POSITIVE
)

net_impact = (
    prevented_fraud_value
    - false_negative_cost
    - false_positive_cost
)


print(
    "\nIllustrative business assumptions:"
)

print(
    f"Cost of missed fraud: "
    f"₦{COST_FALSE_NEGATIVE:,}"
)

print(
    f"Cost of false alert: "
    f"₦{COST_FALSE_POSITIVE:,}"
)

print(
    f"Value of correctly detected fraud: "
    f"₦{VALUE_TRUE_POSITIVE:,}"
)


print(
    "\nEstimated business impact:"
)

print(
    f"Prevented fraud value: "
    f"₦{prevented_fraud_value:,}"
)

print(
    f"False-negative cost: "
    f"₦{false_negative_cost:,}"
)

print(
    f"False-positive cost: "
    f"₦{false_positive_cost:,}"
)

print(
    f"Estimated net impact: "
    f"₦{net_impact:,}"
)


business_impact = pd.DataFrame([
    {
        "metric": "true_negatives",
        "value": true_negative
    },

    {
        "metric": "false_positives",
        "value": false_positive
    },

    {
        "metric": "false_negatives",
        "value": false_negative
    },

    {
        "metric": "true_positives",
        "value": true_positive
    },

    {
        "metric": "prevented_fraud_value",
        "value": prevented_fraud_value
    },

    {
        "metric": "false_negative_cost",
        "value": false_negative_cost
    },

    {
        "metric": "false_positive_cost",
        "value": false_positive_cost
    },

    {
        "metric": "estimated_net_impact",
        "value": net_impact
    }
])


business_impact.to_csv(
    BUSINESS_IMPACT_PATH,
    index=False
)


# ============================================================
# 26. SAVE FINAL METRICS
# ============================================================

final_metrics = pd.DataFrame([
    {
        "model": best_base_model_name,
        "accuracy": final_accuracy,
        "precision": final_precision,
        "recall": final_recall,
        "f1_score": final_f1,
        "roc_auc": final_roc_auc
    }
])


final_metrics.to_csv(
    FINAL_METRICS_PATH,
    index=False
)


# ============================================================
# 27. SAVE MODEL AND ARTIFACTS
# ============================================================

print("\n" + "=" * 75)
print("12. SAVING MODEL AND ARTIFACTS")
print("=" * 75)


joblib.dump(
    best_model,
    FINAL_MODEL_PATH
)


joblib.dump(
    best_model.named_steps[
        "preprocessor"
    ],
    PREPROCESSOR_PATH
)


feature_schema = {
    "target": TARGET,
    "features": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}


joblib.dump(
    feature_schema,
    FEATURE_SCHEMA_PATH
)


print(
    f"Model saved to: "
    f"{FINAL_MODEL_PATH}"
)

print(
    f"Preprocessor saved to: "
    f"{PREPROCESSOR_PATH}"
)

print(
    f"Feature schema saved to: "
    f"{FEATURE_SCHEMA_PATH}"
)


# ============================================================
# 28. PREDICTION FUNCTION
# ============================================================

def predict_transaction(
    transaction_data
):

    """
    Predict fraud for one or more transactions.

    Input can be:
    - A dictionary for one transaction
    - A pandas DataFrame for multiple transactions

    Required features:
    - transaction_amount
    - account_age_days
    - transaction_hour
    - previous_transactions
    - device_type
    - location_match
    - amount_vs_average
    - is_new_device
    """

    # --------------------------------------------------------
    # Convert dictionary to DataFrame
    # --------------------------------------------------------

    if isinstance(
        transaction_data,
        dict
    ):

        transaction_data = pd.DataFrame([
            transaction_data
        ])


    # --------------------------------------------------------
    # Validate input type
    # --------------------------------------------------------

    if not isinstance(
        transaction_data,
        pd.DataFrame
    ):

        raise TypeError(
            "Input must be a dictionary "
            "or pandas DataFrame."
        )


    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    transaction_data = (
        transaction_data
        .copy()
    )

    transaction_data.columns = (
        transaction_data.columns
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in feature_columns
        if column not in transaction_data.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing required features: "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # Select training features
    # --------------------------------------------------------

    input_data = transaction_data[
        feature_columns
    ].copy()


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = best_model.predict(
        input_data
    )


    probabilities = (
        best_model
        .predict_proba(input_data)[:, 1]
    )


    # Confidence means how strongly the
    # model favors either class.

    confidence = np.maximum(
        probabilities,
        1 - probabilities
    )


    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = input_data.copy()


    result[
        "fraud_probability"
    ] = probabilities


    result[
        "confidence"
    ] = confidence


    result[
        "prediction"
    ] = predictions


    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    result[
        "risk_level"
    ] = result[
        "fraud_probability"
    ].apply(
        lambda probability:

        "HIGH"

        if probability >= 0.80

        else (
            "MEDIUM"

            if probability >= 0.50

            else "LOW"
        )
    )


    # --------------------------------------------------------
    # Automated decision
    # --------------------------------------------------------

    result[
        "action"
    ] = result.apply(

        lambda row:

        "AUTO-ACT"

        if (
            row["fraud_probability"]
            >= CONFIDENCE_AUTO_ACTION
        )

        else (

            "HUMAN REVIEW"

            if (
                row["fraud_probability"]
                >= CONFIDENCE_REVIEW
            )

            else "NO ACTION"
        ),

        axis=1
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    result[
        "prediction_timestamp"
    ] = datetime.now().isoformat()


    return result


# ============================================================
# 29. CREATE / LOAD NEW TRANSACTION DATA
# ============================================================

print("\n" + "=" * 75)
print("13. NEW TRANSACTION PREDICTION")
print("=" * 75)


if not os.path.exists(
    NEW_DATA_PATH
):

    print(
        "No new_transactions.csv found."
    )

    print(
        "Creating a demonstration file "
        "using test transactions."
    )

    demo_data = (
        X_test[
            feature_columns
        ]
        .sample(
            n=min(
                20,
                len(X_test)
            ),
            random_state=RANDOM_STATE
        )
        .copy()
    )

    demo_data.to_csv(
        NEW_DATA_PATH,
        index=False
    )

    print(
        f"Created: "
        f"{NEW_DATA_PATH}"
    )

else:

    print(
        f"Using existing file: "
        f"{NEW_DATA_PATH}"
    )


# ============================================================
# 30. RUN BATCH PREDICTIONS
# ============================================================

new_transactions = pd.read_csv(
    NEW_DATA_PATH
)


prediction_results = predict_transaction(
    new_transactions
)


print("\nSample predictions:")

print(
    prediction_results
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 31. SAVE PREDICTIONS
# ============================================================

prediction_results.to_csv(
    PREDICTION_OUTPUT_PATH,
    index=False
)


print(
    f"\nPredictions saved to: "
    f"{PREDICTION_OUTPUT_PATH}"
)


# ============================================================
# 32. PREDICTION SUMMARY
# ============================================================

print("\nPrediction summary:")

print(
    prediction_results[
        "risk_level"
    ].value_counts()
)


# ============================================================
# 33. SQLITE PREDICTION LOG
# ============================================================

print("\n" + "=" * 75)
print("14. ALERT / ACTION SYSTEM")
print("=" * 75)


connection = sqlite3.connect(
    DATABASE_PATH
)


cursor = connection.cursor()


cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS fraud_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_timestamp TEXT,
        fraud_probability REAL,
        confidence REAL,
        prediction INTEGER,
        risk_level TEXT,
        action TEXT
    )
    """
)


for _, row in prediction_results.iterrows():

    cursor.execute(
        """
        INSERT INTO fraud_predictions (
            prediction_timestamp,
            fraud_probability,
            confidence,
            prediction,
            risk_level,
            action
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,

        (
            row[
                "prediction_timestamp"
            ],

            float(
                row[
                    "fraud_probability"
                ]
            ),

            float(
                row[
                    "confidence"
                ]
            ),

            int(
                row[
                    "prediction"
                ]
            ),

            row[
                "risk_level"
            ],

            row[
                "action"
            ]
        )
    )


connection.commit()


# ============================================================
# 34. DISPLAY AUTOMATED ACTION COUNTS
# ============================================================

high_risk_count = int(
    (
        prediction_results[
            "action"
        ]
        == "AUTO-ACT"
    ).sum()
)


review_count = int(
    (
        prediction_results[
            "action"
        ]
        == "HUMAN REVIEW"
    ).sum()
)


no_action_count = int(
    (
        prediction_results[
            "action"
        ]
        == "NO ACTION"
    ).sum()
)


print(
    f"AUTO-ACT transactions: "
    f"{high_risk_count}"
)

print(
    f"HUMAN REVIEW transactions: "
    f"{review_count}"
)

print(
    f"NO ACTION transactions: "
    f"{no_action_count}"
)


# ============================================================
# 35. DISPLAY INDIVIDUAL ACTIONS
# ============================================================

print("\nAutomated decisions:")

for index, row in prediction_results.iterrows():

    transaction_number = index + 1

    probability = (
        row["fraud_probability"]
    )

    action = row["action"]

    risk = row["risk_level"]


    if action == "AUTO-ACT":

        print(
            f"Transaction {transaction_number}: "
            f"HIGH RISK | "
            f"Fraud probability = "
            f"{probability:.1%} | "
            f"ACTION = AUTO-ACT"
        )


    elif action == "HUMAN REVIEW":

        print(
            f"Transaction {transaction_number}: "
            f"MEDIUM RISK | "
            f"Fraud probability = "
            f"{probability:.1%} | "
            f"ACTION = HUMAN REVIEW"
        )


    else:

        print(
            f"Transaction {transaction_number}: "
            f"{risk} RISK | "
            f"Fraud probability = "
            f"{probability:.1%} | "
            f"ACTION = NO ACTION"
        )


connection.close()


print(
    f"\nPrediction log saved to: "
    f"{DATABASE_PATH}"
)


# ============================================================
# 36. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("15. PROJECT SUMMARY")
print("=" * 75)


print(
    f"Dataset: "
    f"{df.shape[0]:,} transactions"
)

print(
    f"Fraud cases: "
    f"{fraud_count:,}"
)

print(
    f"Fraud rate: "
    f"{fraud_rate:.2f}%"
)

print(
    "Models compared: "
    "Logistic Regression, Random Forest, XGBoost"
)

print(
    f"Selected model: "
    f"{best_base_model_name}"
)

print(
    f"Final Accuracy: "
    f"{final_accuracy:.3f}"
)

print(
    f"Final Precision: "
    f"{final_precision:.3f}"
)

print(
    f"Final Recall: "
    f"{final_recall:.3f}"
)

print(
    f"Final F1 Score: "
    f"{final_f1:.3f}"
)

print(
    f"Final ROC-AUC: "
    f"{final_roc_auc:.3f}"
)

print(
    f"Prediction records: "
    f"{len(prediction_results):,}"
)

print(
    f"AUTO-ACT: "
    f"{high_risk_count}"
)

print(
    f"HUMAN REVIEW: "
    f"{review_count}"
)

print(
    f"NO ACTION: "
    f"{no_action_count}"
)


print("\nProject completed successfully.")

print("=" * 75)