import matplotlib.pyplot as plt
import os


def create_charts(customer_df, reviews_df):
    """Generate colorful customer insight charts."""

    os.makedirs("output", exist_ok=True)

    # ----------------------------
    # 1. Sentiment Distribution
    # ----------------------------
    sentiment_counts = customer_df["sentiment"].value_counts()

    plt.figure(figsize=(7, 5))
    sentiment_counts.plot(
        kind="bar",
        color=["#2E86DE", "#E74C3C", "#F1C40F"]
    )
    plt.title("Customer Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("output/sentiment_distribution.png")
    plt.close()

    # ----------------------------
    # 2. Rating Distribution
    # ----------------------------
    rating_counts = reviews_df["Rating"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    rating_counts.plot(
        kind="bar",
        color=["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB"]
    )
    plt.title("Customer Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("output/rating_distribution.png")
    plt.close()

    # ----------------------------
    # 3. Customers by Region
    # ----------------------------
    region_counts = customer_df["region"].value_counts()

    plt.figure(figsize=(8, 5))
    region_counts.plot(
        kind="bar",
        color="#9B59B6"
    )
    plt.title("Customers by Region")
    plt.xlabel("Region")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("output/customers_by_region.png")
    plt.close()

    # ----------------------------
    # 4. Issue Resolution
    # ----------------------------
    resolution_counts = customer_df["issue_resolved"].value_counts()

    plt.figure(figsize=(7, 5))
    resolution_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        colors=["#2ECC71", "#E74C3C", "#95A5A6"]
    )
    plt.title("Issue Resolution Rate")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("output/issue_resolution.png")
    plt.close()

    # ----------------------------
    # 5. Response Time
    # ----------------------------
    plt.figure(figsize=(7, 5))
    plt.hist(
        customer_df["response_time_hours"],
        bins=10,
        color="#3498DB",
        edgecolor="black"
    )
    plt.title("Customer Response Time Distribution")
    plt.xlabel("Response Time (Hours)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig("output/response_time_distribution.png")
    plt.close()

    print("✅ Customer insight charts created successfully!")