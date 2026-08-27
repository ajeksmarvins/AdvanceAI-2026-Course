# ============================================================
# NIA26AI021 - WEEK 6 - VID 3
# MODEL DEPLOYMENT PIPELINE
# Loan Approval Prediction
# ============================================================

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np


# ============================================================
# 1. FILE SETTINGS
# ============================================================

MODEL_FILE = "NIA26AI021_Week6_Vid2_Best_RandomForest.joblib"
DATA_FILE = "loan_approval_dataset.csv"

NEW_DATA_FILE = "NIA26AI021_Week6_Vid3_New_Loan_Records.csv"
RESULT_FILE = "NIA26AI021_Week6_Vid3_Prediction_Results.csv"
DATABASE_FILE = "NIA26AI021_Week6_Vid3_Predictions.db"


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 60)
print("LOAN APPROVAL MODEL DEPLOYMENT")
print("=" * 60)

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")


# ============================================================
# 3. PREPARE FEATURE STRUCTURE
# ============================================================

print("\nPreparing feature structure...")

reference_df = pd.read_csv(DATA_FILE)

# Clean column names
reference_df.columns = reference_df.columns.str.strip()

# Clean text values
for column in reference_df.select_dtypes(include="object").columns:
    reference_df[column] = reference_df[column].str.strip()


# Remove target column
feature_df = reference_df.drop(
    columns=["loan_status"],
    errors="ignore"
)

# Remove ID column if present
feature_df = feature_df.drop(
    columns=["loan_id"],
    errors="ignore"
)

# Convert categorical variables
feature_df = pd.get_dummies(
    feature_df,
    drop_first=True
)

feature_columns = feature_df.columns.tolist()

print(f"Number of model features: {len(feature_columns)}")


# ============================================================
# 4. PREPARE NEW DATA
# ============================================================

def prepare_features(new_data):
    """
    Prepare new loan records so they match the
    feature structure used when training the model.
    """

    data = new_data.copy()

    # Remove target if accidentally included
    data = data.drop(
        columns=["loan_status"],
        errors="ignore"
    )

    # Remove loan ID if present
    data = data.drop(
        columns=["loan_id"],
        errors="ignore"
    )

    # Clean text values
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.strip()

    # Convert categorical variables
    data = pd.get_dummies(
        data,
        drop_first=True
    )

    # Make sure new data has exactly the same columns
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    data = data.astype(float)

    return data


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_loan(data):
    """
    Predict loan approval for new records.

    Returns:
        prediction
        probability
        risk level
        action
    """

    prepared_data = prepare_features(data)

    predictions = model.predict(prepared_data)

    probabilities = model.predict_proba(
        prepared_data
    )[:, 1]

    results = []

    for prediction, probability in zip(
        predictions,
        probabilities
    ):

        # Probability of approval
        approval_probability = probability

        if approval_probability >= 0.70:

            risk_level = "LOW"

            action = "APPROVE_RECOMMENDATION"

        elif approval_probability >= 0.40:

            risk_level = "MEDIUM"

            action = "MANUAL_REVIEW"

        else:

            risk_level = "HIGH"

            action = "REJECT_RECOMMENDATION"

        results.append({
            "prediction": int(prediction),
            "approval_probability": round(
                approval_probability,
                4
            ),
            "risk_level": risk_level,
            "action": action
        })

    return pd.DataFrame(results)


# ============================================================
# 6. CREATE TEST INPUT DATA IF NEEDED
# ============================================================

if not os.path.exists(NEW_DATA_FILE):

    print("\nNew loan input file not found.")

    print(
        "Creating a test input file from "
        "existing loan records..."
    )

    sample_data = reference_df.drop(
        columns=["loan_status"],
        errors="ignore"
    ).copy()

    sample_data = sample_data.drop(
        columns=["loan_id"],
        errors="ignore"
    )

    # Take 5 records for demonstration
    sample_data = sample_data.head(5)

    sample_data.to_csv(
        NEW_DATA_FILE,
        index=False
    )

    print(
        f"Test input created: {NEW_DATA_FILE}"
    )


# ============================================================
# 7. LOAD NEW RECORDS
# ============================================================

print("\n" + "=" * 60)
print("LOADING NEW LOAN RECORDS")
print("=" * 60)

new_loans = pd.read_csv(
    NEW_DATA_FILE
)

print(
    f"New records loaded: {len(new_loans)}"
)


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

prediction_results = predict_loan(
    new_loans
)


# ============================================================
# 9. COMBINE INPUT + RESULTS
# ============================================================

final_results = pd.concat(
    [
        new_loans.reset_index(drop=True),
        prediction_results.reset_index(drop=True)
    ],
    axis=1
)


# Convert prediction number to readable result
final_results["prediction_label"] = (
    final_results["prediction"]
    .map({
        1: "APPROVED",
        0: "REJECTED"
    })
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULTS")
print("=" * 60)

display_columns = [
    "prediction_label",
    "approval_probability",
    "risk_level",
    "action"
]

print(
    final_results[display_columns].to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE RESULTS TO CSV
# ============================================================

final_results.to_csv(
    RESULT_FILE,
    index=False
)

print(
    f"\nPrediction results saved to: "
    f"{RESULT_FILE}"
)


# ============================================================
# 12. DATABASE LOGGING
# ============================================================

print("\n" + "=" * 60)
print("DATABASE LOGGING")
print("=" * 60)

connection = sqlite3.connect(
    DATABASE_FILE
)

cursor = connection.cursor()


# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction INTEGER,
    prediction_label TEXT,
    approval_probability REAL,
    risk_level TEXT,
    action TEXT
)
""")


# Insert predictions
for _, row in final_results.iterrows():

    cursor.execute(
        """
        INSERT INTO predictions (
            prediction,
            prediction_label,
            approval_probability,
            risk_level,
            action
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(row["prediction"]),
            row["prediction_label"],
            float(row["approval_probability"]),
            row["risk_level"],
            row["action"]
        )
    )


connection.commit()


# ============================================================
# 13. VERIFY DATABASE RECORDS
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM predictions"
)

record_count = cursor.fetchone()[0]

print(
    f"Database records: {record_count}"
)


# Show recent records
cursor.execute("""
SELECT
    prediction_label,
    approval_probability,
    risk_level,
    action
FROM predictions
ORDER BY id DESC
LIMIT 5
""")

recent_records = cursor.fetchall()

print("\nRecent database records:")

for record in recent_records:
    print(record)


connection.close()


# ============================================================
# 14. CHECK HIGH-RISK RECORDS
# ============================================================

high_risk_records = final_results[
    final_results["risk_level"] == "HIGH"
]

print("\n" + "=" * 60)
print("RISK CHECK")
print("=" * 60)

print(
    f"High-risk records detected: "
    f"{len(high_risk_records)}"
)

if len(high_risk_records) > 0:

    print(
        "High-risk loan applications were flagged "
        "for rejection recommendation."
    )

else:

    print(
        "No high-risk loan applications were detected."
    )


# ============================================================
# 15. FINAL DEPLOYMENT CHECK
# ============================================================

print("\n" + "=" * 60)
print("END-TO-END DEPLOYMENT CHECK")
print("=" * 60)

print("✓ Model loaded")
print("✓ New data loaded")
print("✓ Predictions generated")
print("✓ Probabilities calculated")
print("✓ Risk levels assigned")
print("✓ Actions assigned")
print("✓ Results saved to CSV")
print("✓ Predictions logged to database")
print("✓ Database records verified")

print("\nVid 3 deployment pipeline completed successfully!")