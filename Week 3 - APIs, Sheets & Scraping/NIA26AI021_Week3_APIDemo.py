import requests
import pandas as pd
import yfinance as yf

# -------------------------------
# API 1 - CoinGecko (Crypto Data)
# -------------------------------

crypto_url = "https://api.coingecko.com/api/v3/coins/markets"

params = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 10,
    "page": 1,
    "sparkline": False
}

try:
    response = requests.get(crypto_url, params=params, timeout=10)
    response.raise_for_status()

    crypto_data = response.json()

    crypto_df = pd.DataFrame(crypto_data)[[
        "name",
        "symbol",
        "current_price",
        "market_cap"
    ]]

    crypto_df["Type"] = "Cryptocurrency"

except requests.exceptions.RequestException as e:
    print("CoinGecko API Error:", e)
    crypto_df = pd.DataFrame()

# -------------------------------
# API 2 - Yahoo Finance (Stocks)
# -------------------------------

stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

stock_data = []

for stock in stocks:
    try:
        ticker = yf.Ticker(stock)
        info = ticker.info

        stock_data.append({
            "name": info.get("shortName"),
            "symbol": stock,
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "Type": "Stock"
        })

    except Exception as e:
        print(f"Error fetching {stock}: {e}")

stock_df = pd.DataFrame(stock_data)

# -------------------------------
# Merge both APIs
# -------------------------------

combined_df = pd.concat([crypto_df, stock_df], ignore_index=True)

combined_df.to_csv("combined_market_data.csv", index=False)

print("\nCombined Dataset")
print(combined_df)

print("\nData saved successfully!")