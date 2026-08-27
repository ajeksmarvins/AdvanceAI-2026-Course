import gspread
import pandas as pd
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials

# Google API scope
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Authenticate
creds = Credentials.from_service_account_file(
    "service_account.json",
    scopes=scope
)

client = gspread.authorize(creds)

# Open Google Sheet
sheet = client.open("Week3 Google Sheets").sheet1

# Read data
df = get_as_dataframe(sheet, evaluate_formulas=True).dropna(how="all")

print("Original Data")
print(df)

# --------------------
# Data Transformation
# --------------------

df["Grade"] = df["Score"].apply(
    lambda x: "A" if x >= 90 else
              "B" if x >= 80 else
              "C"
)

average_score = df["Score"].mean()

summary = pd.DataFrame({
    "Metric": ["Total Students", "Average Score"],
    "Value": [len(df), round(average_score, 2)]
})

# Update first sheet
sheet.clear()
set_with_dataframe(sheet, df)

# Create dashboard sheet (Bonus)
try:
    dashboard = client.open("Week3 Google Sheets").worksheet("Dashboard")
except:
    dashboard = client.open("Week3 Google Sheets").add_worksheet(
        title="Dashboard",
        rows=20,
        cols=5
    )

dashboard.clear()
set_with_dataframe(dashboard, summary)

print("\nGoogle Sheet updated successfully!")