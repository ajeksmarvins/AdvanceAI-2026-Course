# ============================================================
# NIA26AI021 - WEEK 6
# FRAUD DETECTION - VID 3
# Deployment-Ready Fraud Detection Pipeline
# ============================================================

import os
import sqlite3
import joblib
import pandas as pd


# ============================================================
# 1. FILE SETTINGS
# ============================================================

MODEL_FILE = "NIA26AI021_Week6_Fraud_Best_XGBoost.joblib"
DATASET_FILE = "credit_card_fraud_10k.csv"

NEW_DATA_FILE = (
    "NIA26AI021_Week6_Fraud_Vid3_New_Transactions.csv"
)

RESULTS_FILE = (
    "NIA26AI021_Week6_Fraud_Vid3_Prediction_Results.csv"
)

DATABASE_FILE = (
    "NIA26AI021_Week6_Fraud_Vid3_Predictions.db"
)


# ============================================================
# 2. LOAD SAVED MODEL
# ============================================================

print("=" * 60)
print("FRAUD DETECTION DEPLOYMENT PIPELINE")
print("=" * 60)

print("\nLoading saved model...")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# 3. CREATE NEW TRANSACTION DATA
# ============================================================

print("\n" + "=" * 60)
print("PREPARING NEW TRANSACTIONS")
print("=" * 60)

if not os.path.exists(NEW_DATA_FILE):

    source_data = pd.read_csv(DATASET_FILE)

    # Remove the target because new transactions
    # should not contain the actual fraud label.

    new_data = source_data.drop(
        columns=["is_fraud"],
        errors="ignore"
    ).copy()

    # Take five transactions as test records.

    new_data = new_data.head(5).copy()

    # Give them new transaction IDs.

    new_data["transaction_id"] = [
        "NEW_TXN_001",
        "NEW_TXN_002",
        "NEW_TXN_003",
        "NEW_TXN_004",
        "NEW_TXN_005"
    ]

    new_data.to_csv(
        NEW_DATA_FILE,
        index=False
    )

    print(
        f"New transaction file created:"
        f"\n{NEW_DATA_FILE}"
    )

else:

    new_data = pd.read_csv(
        NEW_DATA_FILE
    )

    print(
        f"Existing transaction file loaded:"
        f"\n{NEW_DATA_FILE}"
    )


print("\nNew transactions:")
print(new_data.to_string(index=False))


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

transaction_ids = new_data["transaction_id"].copy()

prediction_data = new_data.drop(
    columns=["transaction_id"],
    errors="ignore"
)


# ============================================================
# 5. GENERATE FRAUD PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

fraud_probabilities = model.predict_proba(
    prediction_data
)[:, 1]

predictions = (
    fraud_probabilities >= 0.50
).astype(int)


# ============================================================
# 6. SMART ALERT FUNCTION
# ============================================================

def generate_smart_alert(probability, transaction_id):

    if probability >= 0.80:
        return {
            "alert": "HIGH FRAUD RISK",
            "action": "AUTO_BLOCK",
            "message": (
                f"Immediate fraud alert for "
                f"{transaction_id}"
            )
        }

    elif probability >= 0.50:
        return {
            "alert": "MEDIUM FRAUD RISK",
            "action": "FLAG_FOR_REVIEW",
            "message": (
                f"Review required for "
                f"{transaction_id}"
            )
        }

    else:
        return {
            "alert": "NO_ALERT",
            "action": "ALLOW",
            "message": (
                f"No alert for "
                f"{transaction_id}"
            )
        }


# ============================================================
# 7. ASSIGN RISK LEVEL
# ============================================================

def assign_risk(probability):

    if probability >= 0.80:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


def assign_action(probability):

    if probability >= 0.80:
        return "AUTO_BLOCK"

    elif probability >= 0.50:
        return "FLAG_FOR_REVIEW"

    else:
        return "ALLOW"


risk_levels = [
    assign_risk(probability)
    for probability in fraud_probabilities
]

smart_alerts = [
    generate_smart_alert(
        probability,
        transaction_id
    )
    for probability, transaction_id
    in zip(
        fraud_probabilities,
        transaction_ids
    )
]

alerts = [
    alert["alert"]
    for alert in smart_alerts
]

actions = [
    alert["action"]
    for alert in smart_alerts
]


# ============================================================
# 8. CREATE PREDICTION RESULTS
# ============================================================

results = new_data.copy()

results["fraud_probability"] = fraud_probabilities

results["prediction"] = predictions

results["risk_level"] = risk_levels

results["smart_alert"] = alerts

results["action"] = actions

# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FRAUD DETECTION RESULTS")
print("=" * 60)

display_columns = [
    "transaction_id",
    "fraud_probability",
    "prediction",
    "risk_level",
    "action"
]

print(
    results[display_columns].to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE RESULTS TO CSV
# ============================================================

results.to_csv(
    RESULTS_FILE,
    index=False
)

print(
    f"\nPrediction results saved to:"
    f"\n{RESULTS_FILE}"
)


# ============================================================
# 10. DATABASE LOGGING
# ============================================================

print("\n" + "=" * 60)
print("DATABASE LOGGING")
print("=" * 60)

connection = sqlite3.connect(
    DATABASE_FILE
)

cursor = connection.cursor()


cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS fraud_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        fraud_probability REAL,
        prediction INTEGER,
        risk_level TEXT,
        action TEXT
    )
    """
)


for _, row in results.iterrows():

    cursor.execute(
        """
        INSERT INTO fraud_predictions (
            transaction_id,
            fraud_probability,
            prediction,
            risk_level,
            action
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            row["transaction_id"],
            float(row["fraud_probability"]),
            int(row["prediction"]),
            row["risk_level"],
            row["action"]
        )
    )


connection.commit()


# ============================================================
# 11. VERIFY DATABASE RECORDS
# ============================================================

cursor.execute(
    """
    SELECT
        transaction_id,
        fraud_probability,
        risk_level,
        action
    FROM fraud_predictions
    ORDER BY id DESC
    LIMIT 5
    """
)

database_records = cursor.fetchall()

print(
    f"Database records:"
    f" {len(database_records)}"
)

print("\nRecent database records:")

for record in database_records:
    print(record)


connection.close()


# ============================================================
# 12. RISK CHECK
# ============================================================

print("\n" + "=" * 60)
print("RISK CHECK")
print("=" * 60)

high_risk_count = sum(
    risk == "HIGH"
    for risk in risk_levels
)

medium_risk_count = sum(
    risk == "MEDIUM"
    for risk in risk_levels
)

low_risk_count = sum(
    risk == "LOW"
    for risk in risk_levels
)

print(
    f"High-risk records detected: "
    f"{high_risk_count}"
)

print(
    f"Medium-risk records detected: "
    f"{medium_risk_count}"
)

print(
    f"Low-risk records detected: "
    f"{low_risk_count}"
)


if high_risk_count > 0:

    print(
        "\nHigh-risk transactions were "
        "flagged for automatic blocking."
    )

elif medium_risk_count > 0:

    print(
        "\nMedium-risk transactions were "
        "sent for review."
    )

else:

    print(
        "\nNo high-risk transactions detected."
    )


# ============================================================
# 13. ACTION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ACTION SUMMARY")
print("=" * 60)

action_summary = (
    results["action"]
    .value_counts()
)

print(action_summary)


# ============================================================
# 14. END-TO-END DEPLOYMENT CHECK
# ============================================================

print("\n" + "=" * 60)
print("END-TO-END DEPLOYMENT CHECK")
print("=" * 60)

print("✓ Model loaded")
print("✓ New transaction data loaded")
print("✓ Fraud probabilities generated")
print("✓ Risk levels assigned")
print("✓ Actions assigned")
print("✓ Results saved to CSV")
print("✓ Predictions logged to database")
print("✓ Database records verified")


# ============================================================
# 15. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("VID 3 DEPLOYMENT PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)