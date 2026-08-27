import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# WEEK 7 VID 3
# TICKET AUTO-ROUTER
# ============================================================

print("=" * 60)
print("WEEK 7 VID 3 - TICKET AUTO-ROUTER")
print("=" * 60)


# ============================================================
# 1. LOAD SUPPORT TICKETS
# ============================================================

df_tickets = pd.read_csv("support_tickets.csv")

print("\nDataset loaded successfully.")
print(f"Total tickets: {len(df_tickets)}")

print("\nColumns:")
print(df_tickets.columns.tolist())


# Check required columns
required_columns = [
    "subject",
    "description",
    "assigned_team"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df_tickets.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 2. PREPARE TEXT
# ============================================================

df_tickets["subject"] = (
    df_tickets["subject"]
    .fillna("")
    .astype(str)
)

df_tickets["description"] = (
    df_tickets["description"]
    .fillna("")
    .astype(str)
)

df_tickets["full_text"] = (
    df_tickets["subject"]
    + " "
    + df_tickets["description"]
)


print("\nText preparation completed.")


# ============================================================
# 3. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 60)
print("TF-IDF VECTORIZATION")
print("=" * 60)

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words="english"
)

X = vectorizer.fit_transform(
    df_tickets["full_text"]
)

y = df_tickets["assigned_team"]

print(
    f"TF-IDF features created: "
    f"{X.shape[1]}"
)

print("\nTeams:")
print(
    y.value_counts()
)


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

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(
    f"Training tickets: {X_train.shape[0]}"
)

print(
    f"Testing tickets : {X_test.shape[0]}"
)


# ============================================================
# 5. TRAIN RANDOM FOREST ROUTER
# ============================================================

print("\n" + "=" * 60)
print("TRAINING ROUTER")
print("=" * 60)

router = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

router.fit(
    X_train,
    y_train
)

print("Random Forest router trained successfully.")


# ============================================================
# 6. EVALUATE ROUTER
# ============================================================

y_pred = router.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("ROUTING EVALUATION")
print("=" * 60)

print(
    f"Routing accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=router.classes_
)

print(
    pd.DataFrame(
        cm,
        index=router.classes_,
        columns=router.classes_
    )
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=router.classes_
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Ticket Router Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "\nConfusion matrix saved as "
    "'confusion_matrix.png'"
)


# ============================================================
# 8. ANALYSE ROUTING MISTAKES
# ============================================================

print("\n" + "=" * 60)
print("ROUTING MISTAKE ANALYSIS")
print("=" * 60)

error_rows = []

for actual, predicted in zip(
    y_test,
    y_pred
):

    if actual != predicted:

        error_rows.append({
            "Actual": actual,
            "Predicted": predicted
        })


if error_rows:

    errors_df = pd.DataFrame(
        error_rows
    )

    confusion_pairs = (
        errors_df
        .value_counts(
            ["Actual", "Predicted"]
        )
        .reset_index(
            name="Count"
        )
        .sort_values(
            "Count",
            ascending=False
        )
    )

    print(
        "\nMost common routing mistakes:"
    )

    print(
        confusion_pairs.to_string(
            index=False
        )
    )

    confusion_pairs.to_csv(
        "routing_errors.csv",
        index=False
    )

    print(
        "\nRouting errors saved as "
        "'routing_errors.csv'"
    )

else:

    print(
        "\nNo classification errors "
        "were found in the test set."
    )


# ============================================================
# 9. SAVE TRAINED COMPONENTS
# ============================================================

joblib.dump(
    router,
    "ticket_router.joblib"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.joblib"
)

print(
    "\nRouter saved as "
    "'ticket_router.joblib'"
)

print(
    "TF-IDF vectorizer saved as "
    "'tfidf_vectorizer.joblib'"
)


# ============================================================
# 10. PRODUCTION ROUTING FUNCTION
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


def route_ticket(
    subject,
    description
):

    full_text = (
        str(subject)
        + " "
        + str(description)
    )

    features = vectorizer.transform(
        [full_text]
    )

    team = router.predict(
        features
    )[0]

    probabilities = router.predict_proba(
        features
    )[0]

    confidence = probabilities.max()

    needs_review = (
        confidence < CONFIDENCE_THRESHOLD
    )

    result = {
        "assigned_team":
            team
            if not needs_review
            else "General Support",

        "predicted_team":
            team,

        "confidence":
            round(confidence, 4),

        "needs_manual_review":
            needs_review
    }

    if needs_review:

        result["note"] = (
            "Low confidence - sent to "
            "General Support for manual review"
        )

    else:

        result["note"] = (
            "High-confidence automatic routing"
        )

    return result


print("\n" + "=" * 60)
print("CONFIDENCE-BASED ROUTING")
print("=" * 60)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD:.0%}"
)

print(
    "Tickets below the threshold "
    "are sent to General Support."
)


# ============================================================
# 11. TEST WITH FIVE NEW TICKETS
# ============================================================

new_tickets = [
    {
        "ticket_id": "NEW_001",
        "subject":
            "Cannot log in - password reset link not arriving",
        "description":
            "I have tried resetting my password three times today."
    },

    {
        "ticket_id": "NEW_002",
        "subject":
            "Incorrect charge on last invoice",
        "description":
            "My bill shows 249.99 but my plan is 79.99."
    },

    {
        "ticket_id": "NEW_003",
        "subject":
            "Internet connection keeps dropping",
        "description":
            "My connection disconnects every few minutes."
    },

    {
        "ticket_id": "NEW_004",
        "subject":
            "I want to understand my account",
        "description":
            "Please explain the services available on my account."
    },

    {
        "ticket_id": "NEW_005",
        "subject":
            "I have a general question",
        "description":
            "I am not sure which department can help me."
    }
]


print("\n" + "=" * 60)
print("NEW TICKET TESTS")
print("=" * 60)


new_results = []

for ticket in new_tickets:

    result = route_ticket(
        ticket["subject"],
        ticket["description"]
    )

    result["ticket_id"] = (
        ticket["ticket_id"]
    )

    result["subject"] = (
        ticket["subject"]
    )

    new_results.append(
        result
    )

    print(
        f"\n{ticket['ticket_id']}"
    )

    print(
        f"Assigned team: "
        f"{result['assigned_team']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.1f}%"
    )

    print(
        f"Manual review: "
        f"{result['needs_manual_review']}"
    )

    print(
        f"Note: "
        f"{result['note']}"
    )


# ============================================================
# 12. SAVE TEST ROUTING RESULTS
# ============================================================

new_results_df = pd.DataFrame(
    new_results
)

new_results_df.to_csv(
    "new_ticket_routing_results.csv",
    index=False
)

print(
    "\nNew ticket results saved as "
    "'new_ticket_routing_results.csv'"
)


# ============================================================
# 13. BATCH ROUTING FUNCTION
# ============================================================

def run_ticket_management(
    new_tickets_csv
):

    tickets = pd.read_csv(
        new_tickets_csv
    )

    results = []

    for idx, row in tickets.iterrows():

        routing = route_ticket(
            row["subject"],
            row["description"]
        )

        routing["ticket_id"] = (
            row.get("ticket_id", idx)
        )

        results.append(
            routing
        )

    df_results = pd.DataFrame(
        results
    )

    total = len(
        df_results
    )

    manual_review = (
        df_results[
            "needs_manual_review"
        ].sum()
    )

    auto_routed = (
        total - manual_review
    )

    print("\n" + "=" * 60)
    print("BATCH ROUTING RESULTS")
    print("=" * 60)

    print(
        f"Total tickets: {total}"
    )

    print(
        f"Auto-routed: {auto_routed}"
    )

    print(
        f"Manual review: {manual_review}"
    )

    print(
        "\nTeam breakdown:"
    )

    print(
        df_results[
            "assigned_team"
        ].value_counts()
    )

    df_results.to_csv(
        "routing_results.csv",
        index=False
    )

    print(
        "\nBatch results saved as "
        "'routing_results.csv'"
    )

    return df_results


# ============================================================
# 14. PRODUCTION QUESTION
# ============================================================

print("\n" + "=" * 60)
print("PRODUCTION ANALYSIS")
print("=" * 60)

print("""
I would not deploy the router without further validation.

The model should first be tested on a larger and more
representative set of tickets. I would monitor routing
accuracy, category-specific errors and the number of tickets
sent for manual review.

The confidence threshold should also be reviewed using real
business costs. A low-confidence fallback is important
because uncertain tickets should reach a human reviewer
instead of being routed incorrectly.
""")

# ============================================================
# 15. RUN BATCH ROUTING
# ============================================================

batch_results = run_ticket_management(
    "new_tickets.csv"
)

print("\nBatch routing completed successfully.")

# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("VID 3 COMPLETED SUCCESSFULLY!")
print("=" * 60)