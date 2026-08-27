import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

from dotenv import load_dotenv

# ==========================================
# Load Environment Variables
# ==========================================
load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# ==========================================
# Create Email
# ==========================================
msg = EmailMessage()

today = datetime.now().strftime("%d %B %Y")

msg["Subject"] = f"Weekly Sales Performance Report - {today}"
msg["From"] = EMAIL_ADDRESS
msg["To"] = "marvinsjay3@gmail.com"

# ==========================================
# Email Body
# ==========================================
msg.set_content(f"""
Hello,

Please find attached the automated Weekly Sales Performance Report.

Attachments include:
• Sales Report (Excel)
• Sales by Region Chart
• Sales by Product Chart
• Quantity Distribution Chart
• Unit Price Distribution Chart

Report Date:
{today}

This report was generated automatically using Python.

Best regards,
AI Automation Reporting System
""")

# ==========================================
# Attach Excel Report
# ==========================================
with open("Sales_Report.xlsx", "rb") as f:
    msg.add_attachment(
        f.read(),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="Sales_Report.xlsx"
    )

# ==========================================
# Attach Charts
# ==========================================
chart_files = [
    "chart_region.png",
    "chart_product.png",
    "chart_quantity.png",
    "chart_price.png"
]

for chart in chart_files:
    with open(chart, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="png",
            filename=chart
        )

# ==========================================
# Send Email
# ==========================================
print("Connecting to Gmail...")

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(EMAIL_ADDRESS, APP_PASSWORD)
    print("Login successful.")
    smtp.send_message(msg)

print("✅ Email sent successfully!")