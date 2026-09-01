import os
import joblib
import pandas as pd
from datetime import datetime


# ============================================================
# WEEK 8 VID 2
# DEPLOYMENT-READY FRAUD DETECTION SYSTEM
# ============================================================

print("=" * 60)
print("WEEK 8 VID 2 - FRAUD DETECTION DEPLOYMENT")
print("=" * 60)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "credit_card_fraud_10k.csv"
MODELS_DIR = "models"

# Instructor-style confidence threshold
CONFIDENCE_THRESHOLD = 0.85

# Instructor-style batch risk thresholds
HIGH_RISK_THRESHOLD = 0.45
MEDIUM_RISK_THRESHOLD = 0.30

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading training data...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

TARGET = "is_fraud"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )


# Use the same feature-selection logic as Week 8 Vid 1
numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()


feature_columns = [
    column
    for column in numeric_columns
    if column != TARGET
]


# Remove identifier columns
feature_columns = [
    column
    for column in feature_columns
    if "id" not in column.lower()
    and "transaction_id" not in column.lower()
]


if not feature_columns:
    raise ValueError(
        "No usable numerical feature columns were found."
    )


X = df[feature_columns].copy()
y = df[TARGET].copy()


# Handle missing values
X = X.fillna(X.median())


print("\nFeature columns:")

for feature in feature_columns:
    print("-", feature)


# ============================================================
# 4. LOAD BEST MODEL FROM VID 1
# ============================================================

VID1_MODEL = "best_xgboost_week8.joblib"
VID1_FEATURES = "feature_columns_week8.joblib"


if not os.path.exists(VID1_MODEL):
    raise FileNotFoundError(
        f"Could not find {VID1_MODEL}"
    )


if not os.path.exists(VID1_FEATURES):
    raise FileNotFoundError(
        f"Could not find {VID1_FEATURES}"
    )


print("\nLoading best XGBoost model from Vid 1...")

best_model = joblib.load(
    VID1_MODEL
)

saved_features = joblib.load(
    VID1_FEATURES
)


print(
    "Model loaded:",
    type(best_model).__name__
)

print(
    "Saved feature columns:",
    saved_features
)


# ============================================================
# 5. FIT SCALER FOR DEPLOYMENT
# ============================================================

from sklearn.preprocessing import StandardScaler


print("\nPreparing deployment scaler...")

deployment_features = df[saved_features].copy()

deployment_features = deployment_features.fillna(
    deployment_features.median()
)


scaler = StandardScaler()

scaler.fit(
    deployment_features
)


print(
    "Scaler prepared successfully."
)


# ============================================================
# 6. SAVE THREE DEPLOYMENT ARTIFACTS
# ============================================================

classifier_path = os.path.join(
    MODELS_DIR,
    "classifier.pkl"
)

scaler_path = os.path.join(
    MODELS_DIR,
    "scaler.pkl"
)

features_path = os.path.join(
    MODELS_DIR,
    "feature_columns.pkl"
)


joblib.dump(
    best_model,
    classifier_path
)

joblib.dump(
    scaler,
    scaler_path
)

joblib.dump(
    saved_features,
    features_path
)


print("\n" + "=" * 60)
print("DEPLOYMENT ARTIFACTS SAVED")
print("=" * 60)

print(
    f"Model: {classifier_path}"
)

print(
    f"Scaler: {scaler_path}"
)

print(
    f"Features: {features_path}"
)


# ============================================================
# 7. LOAD SAVED DEPLOYMENT ARTIFACTS
# ============================================================

print("\nLoading saved deployment artifacts...")

loaded_model = joblib.load(
    classifier_path
)

loaded_scaler = joblib.load(
    scaler_path
)

loaded_features = joblib.load(
    features_path
)


print(
    "Model loaded:",
    type(loaded_model).__name__
)

print(
    "Scaler loaded:",
    type(loaded_scaler).__name__
)

print(
    "Feature columns:",
    loaded_features
)

print(
    "Everything loaded successfully."
)


# ============================================================
# 8. SINGLE PREDICTION FUNCTION
# ============================================================

def predict_transaction(transaction_data):
    """
    Predict a single transaction.

    Returns:
        prediction
        fraud_probability
        confidence
        risk_level
        recommended_action
        prediction_timestamp
    """

    if not isinstance(
        transaction_data,
        dict
    ):
        raise TypeError(
            "transaction_data must be a dictionary."
        )


    # Check required features
    missing_features = [
        feature
        for feature in loaded_features
        if feature not in transaction_data
    ]


    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(
                missing_features
            )
        )


    # Create DataFrame
    new_data = pd.DataFrame(
        [transaction_data]
    )


    # Keep exact feature order
    new_data = new_data[
        loaded_features
    ].copy()


    # Convert to numeric
    for feature in loaded_features:

        new_data[feature] = pd.to_numeric(
            new_data[feature],
            errors="coerce"
        )


    # Fill missing values with scaler means
    for index, feature in enumerate(
        loaded_features
    ):

        if new_data[feature].isna().any():

            new_data.loc[
                new_data[feature].isna(),
                feature
            ] = loaded_scaler.mean_[index]


    # Scale
    new_data_scaled = (
        loaded_scaler.transform(
            new_data
        )
    )


    # Prediction
    prediction = int(
        loaded_model.predict(
            new_data_scaled
        )[0]
    )


    # Probabilities
    probabilities = (
        loaded_model.predict_proba(
            new_data_scaled
        )[0]
    )


    fraud_probability = float(
        probabilities[1]
    )


    confidence = float(
        max(probabilities)
    )


    # ========================================================
    # INSTRUCTOR-STYLE ACTION LOGIC
    # ========================================================

    if (
        prediction == 1
        and confidence > CONFIDENCE_THRESHOLD
    ):

        action = (
            "AUTO-BLOCK and alert security team"
        )

    elif prediction == 1:

        action = (
            "Flag for manual review"
        )

    else:

        action = (
            "Approve transaction"
        )


    # Risk level
    if fraud_probability >= HIGH_RISK_THRESHOLD:

        risk_level = "HIGH"

    elif fraud_probability >= MEDIUM_RISK_THRESHOLD:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    timestamp = datetime.now().isoformat()


    return {
        "prediction": prediction,
        "fraud_probability": round(
            fraud_probability,
            4
        ),
        "confidence": round(
            confidence,
            4
        ),
        "risk_level": risk_level,
        "recommended_action": action,
        "prediction_timestamp": timestamp
    }


# ============================================================
# 9. SINGLE PREDICTION TEST
# ============================================================

print("\n" + "=" * 60)
print("SINGLE PREDICTION TEST")
print("=" * 60)


# Use an actual transaction from your dataset.
# This guarantees that the feature names match your model.

new_transaction = (
    df[loaded_features]
    .iloc[0]
    .to_dict()
)


single_result = predict_transaction(
    new_transaction
)


print("\nTransaction details:")

for feature in loaded_features:

    print(
        f"{feature}: "
        f"{new_transaction[feature]}"
    )


print("\nPrediction:")

print(
    "Fraud"
    if single_result["prediction"] == 1
    else "LEGITIMATE"
)

print(
    "Fraud Probability:",
    single_result["fraud_probability"]
)

print(
    "Confidence:",
    f"{single_result['confidence'] * 100:.1f}%"
)

print(
    "Risk Level:",
    single_result["risk_level"]
)

print(
    "Action:",
    single_result["recommended_action"]
)

print(
    "Timestamp:",
    single_result["prediction_timestamp"]
)


# ============================================================
# 10. BATCH PREDICTION FUNCTION
# ============================================================

def batch_predict(
    input_csv,
    output_csv
):
    """
    Load a CSV and make predictions for every row.

    Adds:
        prediction
        fraud_probability
        confidence
        risk_level
        prediction_timestamp
        recommended_action
    """

    print("\n" + "=" * 60)
    print("BATCH PREDICTION PIPELINE")
    print("=" * 60)


    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    print("\nLoading new transactions...")

    batch_df = pd.read_csv(
        input_csv
    )


    print(
        "Loaded",
        len(batch_df),
        "records"
    )


    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in loaded_features
        if feature not in batch_df.columns
    ]


    if missing_features:

        raise ValueError(
            "Input CSV is missing features: "
            + ", ".join(
                missing_features
            )
        )


    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    X_batch = batch_df[
        loaded_features
    ].copy()


    # Convert to numeric
    for feature in loaded_features:

        X_batch[feature] = pd.to_numeric(
            X_batch[feature],
            errors="coerce"
        )


    # Fill missing values
    for index, feature in enumerate(
        loaded_features
    ):

        X_batch[feature] = (
            X_batch[feature]
            .fillna(
                loaded_scaler.mean_[index]
            )
        )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    X_batch_scaled = (
        loaded_scaler.transform(
            X_batch
        )
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print(
        "Making predictions..."
    )


    predictions = (
        loaded_model.predict(
            X_batch_scaled
        )
    )


    probabilities = (
        loaded_model.predict_proba(
            X_batch_scaled
        )
    )


    fraud_probabilities = (
        probabilities[:, 1]
    )


    confidence_scores = (
        probabilities.max(
            axis=1
        )
    )


    # --------------------------------------------------------
    # ADD RESULTS
    # --------------------------------------------------------

    batch_df["prediction"] = (
        predictions
    )


    batch_df["fraud_probability"] = (
        fraud_probabilities.round(4)
    )


    batch_df["confidence"] = (
        confidence_scores.round(4)
    )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    def assign_risk_level(
        probability
    ):

        if probability >= HIGH_RISK_THRESHOLD:

            return "HIGH"

        elif probability >= MEDIUM_RISK_THRESHOLD:

            return "MEDIUM"

        else:

            return "LOW"


    batch_df["risk_level"] = (
        batch_df[
            "fraud_probability"
        ].apply(
            assign_risk_level
        )
    )


    # --------------------------------------------------------
    # RECOMMENDED ACTION
    # --------------------------------------------------------

    def assign_action(
        prediction,
        confidence
    ):

        if (
            prediction == 1
            and confidence > CONFIDENCE_THRESHOLD
        ):

            return (
                "AUTO-BLOCK and alert security team"
            )

        elif prediction == 1:

            return (
                "Flag for manual review"
            )

        else:

            return (
                "Approve transaction"
            )


    batch_df["recommended_action"] = [
        assign_action(
            prediction,
            confidence
        )

        for prediction, confidence
        in zip(
            predictions,
            confidence_scores
        )
    ]


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    batch_df[
        "prediction_timestamp"
    ] = datetime.now().isoformat()


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    batch_df.to_csv(
        output_csv,
        index=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "-" * 40)
    print("BATCH SUMMARY")
    print("-" * 40)


    high_count = (
        batch_df["risk_level"]
        .eq("HIGH")
        .sum()
    )

    medium_count = (
        batch_df["risk_level"]
        .eq("MEDIUM")
        .sum()
    )

    low_count = (
        batch_df["risk_level"]
        .eq("LOW")
        .sum()
    )


    print(
        "HIGH risk:",
        high_count
    )

    print(
        "MEDIUM risk:",
        medium_count
    )

    print(
        "LOW risk:",
        low_count
    )


    print(
        "\nBatch results saved as:",
        output_csv
    )


    return batch_df


# ============================================================
# 11. CREATE SAMPLE BATCH FILE
# ============================================================

sample_input = "new_transactions.csv"

sample_output = "batch_predictions.csv"


# Use 50 records when possible,
# similar to the instructor's demo.

sample_size = min(
    50,
    len(df)
)


sample_transactions = (
    df[loaded_features]
    .sample(
        n=sample_size,
        random_state=42
    )
    .copy()
)


sample_transactions.to_csv(
    sample_input,
    index=False
)


print(
    f"\nSample batch file created: "
    f"{sample_input}"
)

print(
    f"Sample records: "
    f"{sample_size}"
)


# ============================================================
# 12. RUN BATCH PIPELINE
# ============================================================

batch_results = batch_predict(
    sample_input,
    sample_output
)


# ============================================================
# 13. SHOW SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE BATCH RESULTS")
print("=" * 60)


display_columns = (
    loaded_features
    + [
        "prediction",
        "fraud_probability",
        "confidence",
        "risk_level",
        "recommended_action"
    ]
)


print(
    batch_results[
        display_columns
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# 14. DEPLOYMENT FILE CHECK
# ============================================================

print("\n" + "=" * 60)
print("DEPLOYMENT FILE CHECK")
print("=" * 60)


required_files = [
    classifier_path,
    scaler_path,
    features_path,
    sample_input,
    sample_output
]


all_files_exist = True


for file_path in required_files:

    exists = os.path.exists(
        file_path
    )

    print(
        f"{file_path}: "
        f"{'OK' if exists else 'MISSING'}"
    )

    if not exists:

        all_files_exist = False


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

if all_files_exist:

    print(
        "VID 2 COMPLETED SUCCESSFULLY!"
    )

else:

    print(
        "VID 2 COMPLETED WITH MISSING FILES."
    )

print("=" * 60)