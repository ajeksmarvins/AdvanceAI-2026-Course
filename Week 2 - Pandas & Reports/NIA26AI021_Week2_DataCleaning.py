import pandas as pd

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("messy_sales.csv")

# ==========================
# BEFORE CLEANING
# ==========================
original_rows = len(df)

print("=" * 50)
print("INSPECTION REPORT")
print("=" * 50)

print("\nSHAPE:")
print(df.shape)

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

# ==========================
# REMOVE DUPLICATES
# ==========================
duplicates_removed = df.duplicated().sum()
df = df.drop_duplicates()

# ==========================
# CLEAN DATES
# ==========================
df["date"] = pd.to_datetime(df["date"], errors="coerce")

invalid_dates = df["date"].isna().sum()

# ==========================
# CLEAN PRODUCT NAMES
# ==========================
before_products = df["product"].astype(str)

df["product"] = (
    df["product"]
    .astype(str)
    .str.strip()
    .str.title()
)

changed_products = (before_products != df["product"]).sum()

# ==========================
# CLEAN QUANTITY
# ==========================
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

missing_quantity = df["quantity"].isna().sum()

df["quantity"] = df["quantity"].fillna(df["quantity"].median())

# ==========================
# CLEAN REGION
# ==========================
missing_region = df["region"].isna().sum()

df["region"] = df["region"].fillna("Unknown")

# ==========================
# CLEAN UNIT PRICE
# ==========================
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

invalid_price = (df["unit_price"] <= 0).sum()

product_medians = (
    df[df["unit_price"] > 0]
    .groupby("product")["unit_price"]
    .median()
)

def fix_price(row):
    price = row["unit_price"]

    if pd.isna(price) or price <= 0:
        if row["product"] in product_medians:
            return product_medians[row["product"]]
        else:
            return df[df["unit_price"] > 0]["unit_price"].median()

    return price

df["unit_price"] = df.apply(fix_price, axis=1)

# ==========================
# CALCULATED COLUMN
# ==========================
df["total_sales"] = df["quantity"] * df["unit_price"]

# ==========================
# SUMMARY
# ==========================
summary = (
    df.groupby("region")["total_sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOTAL SALES BY REGION")
print(summary)

# ==========================
# SAVE FILES
# ==========================
df.to_csv("sales_data_CLEANED.csv", index=False)
df.to_excel("sales_data_CLEANED.xlsx", index=False)

# ==========================
# AFTER CLEANING
# ==========================
print("\n" + "=" * 50)
print("CLEANING SUMMARY")
print("=" * 50)

print(f"Rows before cleaning: {original_rows}")
print(f"Rows after cleaning : {len(df)}")
print(f"Duplicate rows removed: {duplicates_removed}")
print(f"Missing quantities fixed: {missing_quantity}")
print(f"Missing regions fixed: {missing_region}")
print(f"Invalid dates found: {invalid_dates}")
print(f"Zero/Negative prices fixed: {invalid_price}")
print(f"Product names standardized: {changed_products}")

print("\nFiles created:")
print("✔ sales_data_CLEANED.csv")
print("✔ sales_data_CLEANED.xlsx")