import smtplib
from email.message import EmailMessage
from config import EMAIL_ADDRESS, APP_PASSWORD, RECIPIENTS

def send_email():
    """
    Sends the generated customer insights via email.
    """

    try:
        msg = EmailMessage()

        msg["Subject"] = "Customer Insights Automated Report"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = ", ".join(RECIPIENTS)
        msg.set_content(
    "Hello,\n\n"
    "Please find attached the latest Customer Insights automated report.\n\n"
    "The report includes customer sentiment, review analysis, regional insights, "
    "issue resolution, response-time analysis, and visualizations.\n\n"
    "Regards,\n"
    "Automated Customer Insights System"
)

        # Attach Excel Report
        with open("output/Customer_Insights_Report.xlsx", "rb") as file:
            msg.add_attachment(
                file.read(),
                maintype="application",
                subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename="Customer_Insights_Report.xlsx"
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, APP_PASSWORD)
            smtp.send_message(msg)

        print("✅ Email sent successfully!")

    except Exception as e:
        print(f"❌ Email sending failed: {e}")