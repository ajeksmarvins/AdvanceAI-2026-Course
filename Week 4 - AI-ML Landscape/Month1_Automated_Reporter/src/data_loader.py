import pandas as pd
import os


def load_customer_sentiment():
    """Load the customer sentiment dataset."""
    
    path = os.path.join("data", "Customer_Sentiment.csv")

    try:
        df = pd.read_csv(path)
        print("✅ Customer sentiment data loaded successfully.")
        print(f"Rows loaded: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        return df

    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None

    except Exception as e:
        print(f"❌ Error loading customer sentiment data: {e}")
        return None


def load_reviews_data():
    """Load the web-scraped reviews dataset."""
    
    path = os.path.join("data", "reviews_data.csv")

    try:
        df = pd.read_csv(path)
        print("✅ Reviews data loaded successfully.")
        print(f"Rows loaded: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        return df

    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None

    except Exception as e:
        print(f"❌ Error loading reviews data: {e}")
        return None