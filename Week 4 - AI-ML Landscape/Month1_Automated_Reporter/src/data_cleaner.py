import pandas as pd


def clean_customer_sentiment(df):
    """Clean the customer sentiment dataset."""

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean text columns
    text_columns = [
        "gender",
        "age_group",
        "region",
        "product_category",
        "purchase_channel",
        "platform",
        "review_text",
        "sentiment"
    ]

    for column in text_columns:
        df[column] = df[column].fillna("Unknown")
        df[column] = df[column].astype(str).str.strip()

    # Clean ratings
    df["customer_rating"] = pd.to_numeric(
        df["customer_rating"], errors="coerce"
    )

    df["customer_rating"] = df["customer_rating"].fillna(
        df["customer_rating"].median()
    )

    # Clean response time
    df["response_time_hours"] = pd.to_numeric(
        df["response_time_hours"], errors="coerce"
    )

    df["response_time_hours"] = df["response_time_hours"].fillna(0)

    # Clean yes/no fields
    for column in ["issue_resolved", "complaint_registered"]:
        df[column] = df[column].fillna("Unknown")
        df[column] = df[column].astype(str).str.strip().str.lower()

    print("✅ Customer sentiment data cleaned successfully.")
    print(f"Rows after cleaning: {len(df)}")

    return df


def clean_reviews_data(df):
    """Clean the web-scraped reviews dataset."""

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean text columns
    for column in ["name", "location", "Review"]:
        df[column] = df[column].fillna("Unknown")
        df[column] = df[column].astype(str).str.strip()

    # Clean ratings
    df["Rating"] = pd.to_numeric(
        df["Rating"], errors="coerce"
    )

    df = df[df["Rating"].between(1, 5)]

    # Convert review dates
    df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
    format="mixed"
)

    # Remove rows where the review text is missing
    df = df[df["Review"] != "Unknown"]

    print("✅ Reviews data cleaned successfully.")
    print(f"Rows after cleaning: {len(df)}")

    return df
