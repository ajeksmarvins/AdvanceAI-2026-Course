import os
from dotenv import load_dotenv

load_dotenv()

# ==========================
# PATHS
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(BASE_DIR, "data")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
LOG_FOLDER = os.path.join(BASE_DIR, "log")

# ==========================
# FILES
# ==========================

CUSTOMER_SENTIMENT_FILE = os.path.join(
    DATA_FOLDER, "Customer_Sentiment.csv"
)

REVIEWS_FILE = os.path.join(
    DATA_FOLDER, "reviews_data.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_FOLDER, "Customer_Insights_Report.xlsx"
)

# ==========================
# EMAIL SETTINGS
# ==========================

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
APP_PASSWORD = os.getenv("APP_PASSWORD")

RECIPIENTS = [
    email.strip()
    for email in os.getenv("RECIPIENTS", "").split(",")
    if email.strip()
]