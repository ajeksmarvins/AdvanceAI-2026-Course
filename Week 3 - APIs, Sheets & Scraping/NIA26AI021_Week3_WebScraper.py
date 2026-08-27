"""
Week 3 Practice 3
Website: https://books.toscrape.com
Purpose:
Scrape book information from a public website, save to CSV,
and combine with Google Sheets data (Bonus Challenge).
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe

books = []

try:
    # Scrape first 3 pages
    for page in range(1, 4):

        if page == 1:
            url = "https://books.toscrape.com/"
        else:
            url = f"https://books.toscrape.com/catalogue/page-{page}.html"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        for book in soup.select("article.product_pod"):

            title = book.h3.a["title"]

            price = book.select_one(".price_color").text

            availability = book.select_one(".availability").text.strip()

            books.append({
                "Title": title,
                "Price": price,
                "Availability": availability
            })

        print(f"Page {page} completed.")
        time.sleep(1)

except Exception as e:
    print("Scraping Error:", e)

# Save scraped data
books_df = pd.DataFrame(books)
books_df.to_csv("books_data.csv", index=False)

print("Books saved successfully!")

# ---------------- BONUS ----------------

try:

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_file(
        "service_account.json",
        scopes=scope
    )

    client = gspread.authorize(creds)

    sheet = client.open("Week3 Google Sheets").sheet1

    students_df = get_as_dataframe(sheet).dropna(how="all")

    print("\nGoogle Sheet Data:")
    print(students_df)

    print("\nBooks Scraped:", len(books_df))
    print("Students Loaded:", len(students_df))

except Exception as e:
    print("Google Sheets Error:", e)

print("\nWeek 3 Practice 3 Completed Successfully!")