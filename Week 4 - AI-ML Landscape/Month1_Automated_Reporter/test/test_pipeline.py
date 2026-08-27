import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_cleaner import clean_customer_sentiment
from src.analyzer import analyze_customer_sentiment

def test_customer_sentiment_cleaning():
    data = pd.DataFrame({
        "customer_id": [1, 2, 2],
        "gender": ["female", "male", "female"],
        "age_group": ["18-25", "26-35", "18-25"],
        "region": ["West", None, "East"],
        "product_category": ["Electronics", "Clothing", "Food"],
        "purchase_channel": ["Online", "Store", "Online"],
        "platform": ["Web", "Mobile", "Web"],
        "customer_rating": [5, 3, 4],
        "review_text": ["Great", "Okay", "Good"],
        "sentiment": ["positive", "negative", "neutral"],
        "response_time_hours": [10, 20, 30],
        "issue_resolved": ["yes", "no", "yes"],
        "complaint_registered": ["no", "yes", "no"]
    })

    cleaned = clean_customer_sentiment(data)

    assert len(cleaned) <= len(data)
    assert cleaned["region"].isna().sum() == 0
    assert cleaned["gender"].isna().sum() == 0

def test_customer_sentiment_analysis():
    data = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "region": ["West", "East", "West"],
        "sentiment": ["positive", "negative", "positive"],
        "response_time_hours": [10, 20, 30],
        "issue_resolved": ["yes", "no", "yes"],
    })

    summary = analyze_customer_sentiment(data)

    assert "total_customers" in summary
    assert "average_response_time" in summary
    assert "sentiment_distribution" in summary
    assert "top_region" in summary
    assert "resolution_rate" in summary