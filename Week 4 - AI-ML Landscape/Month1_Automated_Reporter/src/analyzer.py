import pandas as pd


def analyze_customer_sentiment(df):
    """Analyze customer sentiment data and return key insights."""

    summary = {
        "total_customers": len(df),
        "average_response_time": round(
            df["response_time_hours"].mean(), 2
        ),
        "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        "top_region": df["region"].value_counts().idxmax(),
        "resolution_rate": round(
            (df["issue_resolved"].str.lower() == "yes").mean() * 100, 2
        ),
    }

    print("✅ Customer sentiment analysis completed.")

    return summary


def analyze_reviews(df):
    """Analyze the web-scraped customer reviews."""

    summary = {
        "total_reviews": len(df),
        "average_rating": round(df["Rating"].mean(), 2),
        "rating_distribution": df["Rating"].value_counts().sort_index().to_dict(),
        "top_location": df["location"].value_counts().idxmax(),
    }

    print("✅ Reviews analysis completed.")

    return summary