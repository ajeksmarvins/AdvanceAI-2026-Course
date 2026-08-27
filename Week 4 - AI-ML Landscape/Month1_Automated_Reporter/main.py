from src.data_loader import load_customer_sentiment, load_reviews_data
from src.data_cleaner import clean_customer_sentiment, clean_reviews_data
from src.analyzer import analyze_customer_sentiment, analyze_reviews
from src.visualizer import create_charts
from src.report_generator import create_report
from src.email_sender import send_email

import logging
import os


# ==========================
# Logging Setup
# ==========================

os.makedirs("log", exist_ok=True)

logging.basicConfig(
    filename="log/customer_insights.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================
# Start Pipeline
# ==========================

logger.info("Customer Insights pipeline started.")


# ==========================
# Load Data
# ==========================

customer_df = load_customer_sentiment()
reviews_df = load_reviews_data()

if customer_df is None or reviews_df is None:
    logger.error("Pipeline stopped because one or more datasets could not be loaded.")
    print("❌ Pipeline stopped: one or more datasets could not be loaded.")
    raise SystemExit(1)


# ==========================
# Clean Data
# ==========================

if customer_df is not None:
    customer_df = clean_customer_sentiment(customer_df)

if reviews_df is not None:
    reviews_df = clean_reviews_data(reviews_df)


# ==========================
# Analyze Data
# ==========================

customer_summary = analyze_customer_sentiment(customer_df)
reviews_summary = analyze_reviews(reviews_df)

logger.info("Customer and review analysis completed successfully.")


# ==========================
# Display Insights
# ==========================

print("\n========== CUSTOMER INSIGHTS ==========")

for key, value in customer_summary.items():
    print(f"{key}: {value}")


print("\n========== REVIEW INSIGHTS ==========")

for key, value in reviews_summary.items():
    print(f"{key}: {value}")


# ==========================
# Create Charts
# ==========================

create_charts(customer_df, reviews_df)


# ==========================
# Create Excel Report
# ==========================

create_report(
    customer_df,
    reviews_df,
    customer_summary,
    reviews_summary
)

logger.info("Excel report created successfully.")


# ==========================
# Send Email
# ==========================

send_email()

logger.info("Email sent successfully.")
logger.info("Customer Insights pipeline completed successfully.")


print("\n✅ Customer Insights pipeline completed successfully.")