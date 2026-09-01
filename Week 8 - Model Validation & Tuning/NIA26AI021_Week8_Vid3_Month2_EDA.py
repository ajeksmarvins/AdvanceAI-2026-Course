# ============================================================
# NIA26AI021 - WEEK 8 VID 3
# MONTH 2 PROJECT - EXPLORATORY DATA ANALYSIS
# Project: Credit Card Fraud Detection
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("WEEK 8 VID 3 - MONTH 2 PROJECT")
print("CREDIT CARD FRAUD DETECTION")
print("=" * 70)

print("\nProject Topic:")
print("Credit Card Fraud Detection")

print("\nObjective:")
print(
    "Explore transaction data to understand its structure, data quality, "
    "target distribution, and important patterns before building machine "
    "learning models."
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATA_FILE = "credit_card_fraud_10k.csv"
TARGET = "is_fraud"

print("\n" + "=" * 70)
print("1. DATASET LOADING")
print("=" * 70)

try:
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset loaded successfully: {DATA_FILE}")
except FileNotFoundError:
    print(f"ERROR: Could not find '{DATA_FILE}'.")
    print("Make sure the CSV file is in the same folder as this script.")
    raise


# ============================================================
# 3. DATASET SHAPE
# ============================================================

print("\n" + "=" * 70)
print("2. DATASET SHAPE")
print("=" * 70)

rows, columns = df.shape

print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 4. FIRST FIVE ROWS
# ============================================================

print("\n" + "=" * 70)
print("3. FIRST FIVE ROWS")
print("=" * 70)

print(df.head())


# ============================================================
# 5. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("4. COLUMN NAMES")
print("=" * 70)

for column in df.columns:
    print(f"- {column}")


# ============================================================
# 6. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("5. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 7. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("6. DATASET INFORMATION")
print("=" * 70)

df.info()


# ============================================================
# 8. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print("Missing values by column:")
print(missing_values)

total_missing = int(missing_values.sum())

print(f"\nTotal missing values: {total_missing}")

if total_missing == 0:
    print("Result: No missing values were found.")
else:
    print("Result: Missing values were found and should be investigated.")


# ============================================================
# 9. DUPLICATE ROW ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("8. DUPLICATE ROW ANALYSIS")
print("=" * 70)

duplicate_count = int(df.duplicated().sum())

print(f"Number of duplicate rows: {duplicate_count}")

if duplicate_count == 0:
    print("Result: No duplicate rows were found.")
else:
    print("Result: Duplicate rows were found and should be investigated.")


# ============================================================
# 10. TARGET VARIABLE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("9. TARGET VARIABLE ANALYSIS")
print("=" * 70)

if TARGET not in df.columns:
    print(f"ERROR: Target column '{TARGET}' was not found.")
    raise KeyError(TARGET)

print(f"Target variable: {TARGET}")

target_counts = df[TARGET].value_counts().sort_index()

print("\nTarget value counts:")
print(target_counts)

target_percentages = df[TARGET].value_counts(normalize=True).sort_index() * 100

print("\nTarget percentages:")
print(target_percentages.round(2))


# ============================================================
# 11. CLASS IMBALANCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("10. CLASS IMBALANCE ANALYSIS")
print("=" * 70)

legitimate_count = int((df[TARGET] == 0).sum())
fraud_count = int((df[TARGET] == 1).sum())

print(f"Legitimate transactions: {legitimate_count}")
print(f"Fraudulent transactions: {fraud_count}")

if fraud_count > 0:

    imbalance_ratio = legitimate_count / fraud_count

    legitimate_percentage = (
        legitimate_count / len(df)
    ) * 100

    fraud_percentage = (
        fraud_count / len(df)
    ) * 100

    print(
        f"Legitimate transactions: "
        f"{legitimate_percentage:.2f}%"
    )

    print(
        f"Fraudulent transactions: "
        f"{fraud_percentage:.2f}%"
    )

    print(
        f"Legitimate-to-fraud ratio: "
        f"{imbalance_ratio:.2f}:1"
    )

    if imbalance_ratio >= 2:
        print("\nResult: The target variable is imbalanced.")
        print(
            "Fraud cases are much less common than legitimate "
            "transactions."
        )
    else:
        print("\nResult: The target variable is relatively balanced.")

else:
    print("\nNo fraud cases were found in the dataset.")


# ============================================================
# 12. BASIC DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("11. BASIC DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe())


# ============================================================
# 13. NUMERIC COLUMN SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("12. NUMERIC COLUMN SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:

    print(f"\n{column}")

    print(
        f"  Mean    : "
        f"{df[column].mean():.4f}"
    )

    print(
        f"  Median  : "
        f"{df[column].median():.4f}"
    )

    print(
        f"  Minimum : "
        f"{df[column].min():.4f}"
    )

    print(
        f"  Maximum : "
        f"{df[column].max():.4f}"
    )


# ============================================================
# 14. CATEGORICAL DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("13. CATEGORICAL DATA ANALYSIS")
print("=" * 70)

categorical_columns = df.select_dtypes(include="object").columns

if len(categorical_columns) == 0:

    print("No categorical columns found.")

else:

    for column in categorical_columns:

        print(f"\nColumn: {column}")
        print(f"Unique values: {df[column].nunique()}")

        print("Most common values:")
        print(df[column].value_counts().head(10))


# ============================================================
# 15. FRAUD RATE BY MERCHANT CATEGORY
# ============================================================

if "merchant_category" in df.columns:

    print("\n" + "=" * 70)
    print("14. FRAUD RATE BY MERCHANT CATEGORY")
    print("=" * 70)

    fraud_by_category = (
        df.groupby("merchant_category")[TARGET]
        .agg(["count", "sum"])
        .rename(
            columns={
                "count": "transactions",
                "sum": "fraud_cases"
            }
        )
    )

    fraud_by_category["fraud_rate_percent"] = (
        fraud_by_category["fraud_cases"]
        / fraud_by_category["transactions"]
        * 100
    )

    fraud_by_category = fraud_by_category.sort_values(
        "fraud_rate_percent",
        ascending=False
    )

    print(fraud_by_category)


# ============================================================
# 16. FRAUD RATE BY FOREIGN TRANSACTION
# ============================================================

if "foreign_transaction" in df.columns:

    print("\n" + "=" * 70)
    print("15. FRAUD RATE BY FOREIGN TRANSACTION")
    print("=" * 70)

    foreign_analysis = (
        df.groupby("foreign_transaction")[TARGET]
        .agg(["count", "sum"])
        .rename(
            columns={
                "count": "transactions",
                "sum": "fraud_cases"
            }
        )
    )

    foreign_analysis["fraud_rate_percent"] = (
        foreign_analysis["fraud_cases"]
        / foreign_analysis["transactions"]
        * 100
    )

    print(foreign_analysis)


# ============================================================
# 17. FRAUD RATE BY LOCATION MISMATCH
# ============================================================

if "location_mismatch" in df.columns:

    print("\n" + "=" * 70)
    print("16. FRAUD RATE BY LOCATION MISMATCH")
    print("=" * 70)

    location_analysis = (
        df.groupby("location_mismatch")[TARGET]
        .agg(["count", "sum"])
        .rename(
            columns={
                "count": "transactions",
                "sum": "fraud_cases"
            }
        )
    )

    location_analysis["fraud_rate_percent"] = (
        location_analysis["fraud_cases"]
        / location_analysis["transactions"]
        * 100
    )

    print(location_analysis)


# ============================================================
# 18. VISUALIZATION 1 - TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("17. VISUALIZATIONS")
print("=" * 70)

plt.figure(figsize=(7, 5))

df[TARGET].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "vid3_target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 19. VISUALIZATION 2 - TRANSACTION AMOUNT DISTRIBUTION
# ============================================================

if "amount" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["amount"],
        bins=30
    )

    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        "vid3_amount_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 20. VISUALIZATION 3 - FRAUD BY TRANSACTION HOUR
# ============================================================

if "transaction_hour" in df.columns:

    fraud_by_hour = (
        df.groupby("transaction_hour")[TARGET]
        .sum()
    )

    plt.figure(figsize=(9, 5))

    fraud_by_hour.plot(
        kind="line",
        marker="o"
    )

    plt.title("Fraud Cases by Transaction Hour")
    plt.xlabel("Transaction Hour")
    plt.ylabel("Number of Fraud Cases")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "vid3_fraud_by_hour.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 21. VISUALIZATION 4 - FRAUD BY FOREIGN TRANSACTION
# ============================================================

if "foreign_transaction" in df.columns:

    fraud_foreign = (
        df.groupby("foreign_transaction")[TARGET]
        .sum()
    )

    plt.figure(figsize=(7, 5))

    fraud_foreign.plot(
        kind="bar"
    )

    plt.title("Fraud Cases by Foreign Transaction Status")
    plt.xlabel("Foreign Transaction (0 = No, 1 = Yes)")
    plt.ylabel("Number of Fraud Cases")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "vid3_fraud_foreign.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 22. VISUALIZATION 5 - FRAUD BY LOCATION MISMATCH
# ============================================================

if "location_mismatch" in df.columns:

    fraud_location = (
        df.groupby("location_mismatch")[TARGET]
        .sum()
    )

    plt.figure(figsize=(7, 5))

    fraud_location.plot(
        kind="bar"
    )

    plt.title("Fraud Cases by Location Mismatch")
    plt.xlabel("Location Mismatch (0 = No, 1 = Yes)")
    plt.ylabel("Number of Fraud Cases")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "vid3_fraud_location_mismatch.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 23. INITIAL EDA FINDINGS
# ============================================================

print("\n" + "=" * 70)
print("18. INITIAL EDA FINDINGS")
print("=" * 70)

print(
    f"""
1. The dataset contains {rows:,} transaction records and
   {columns} columns for credit card fraud detection.

2. The target variable is '{TARGET}', where:
   0 represents a legitimate transaction.
   1 represents a fraudulent transaction.

3. The dataset contains {legitimate_count:,} legitimate transactions
   ({legitimate_percentage:.2f}%) and {fraud_count:,} fraudulent
   transactions ({fraud_percentage:.2f}%).

4. The legitimate-to-fraud ratio is approximately
   {imbalance_ratio:.2f}:1, showing that the target variable is
   highly imbalanced.

5. Missing values were checked across all columns.
   Total missing values found: {total_missing}.

6. Duplicate records were checked.
   Total duplicate rows found: {duplicate_count}.

7. The dataset contains numeric transaction and cardholder features,
   as well as categorical transaction information.

8. Potentially useful fraud-related variables include transaction
   amount, transaction hour, merchant category, foreign transaction
   status, location mismatch, device trust score, transaction
   velocity, and cardholder age.

9. The EDA results will guide feature preparation, model selection,
   class-imbalance handling, and model evaluation in the next stage.
"""
)


# ============================================================
# 24. PLANNED MACHINE LEARNING ALGORITHMS
# ============================================================

print("\n" + "=" * 70)
print("19. PLANNED MACHINE LEARNING ALGORITHMS")
print("=" * 70)

print(
    """
The following algorithms are planned for comparison:

1. Logistic Regression
   - Provides a simple and interpretable baseline.

2. Random Forest
   - Can capture nonlinear relationships and interactions between
     transaction features.

3. XGBoost
   - A powerful gradient-boosting algorithm suitable for structured
     classification data and commonly effective for fraud detection.

Model performance will be compared using fraud-focused metrics such as:

- Precision
- Recall
- F1 Score
- ROC-AUC

Because the target is imbalanced, accuracy alone will not be used as
the primary model-selection metric.
"""
)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("20. VID 3 COMPLETED")
print("=" * 70)

print("Dataset loaded successfully.")
print("Initial exploratory data analysis completed.")
print("Missing values checked.")
print("Duplicate records checked.")
print("Target distribution examined.")
print("Class imbalance evaluated.")
print("Basic statistics calculated.")
print("Categorical variables analyzed.")
print("Fraud patterns examined.")
print("Visualizations generated.")
print("Initial findings documented.")
print("Machine learning algorithms selected for comparison.")

print("\nGenerated visualization files:")
print("- vid3_target_distribution.png")
print("- vid3_amount_distribution.png")
print("- vid3_fraud_by_hour.png")
print("- vid3_fraud_foreign.png")
print("- vid3_fraud_location_mismatch.png")

print("\nWeek 8 Vid 3 is ready for review.")
print("=" * 70)