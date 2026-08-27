import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# WEEK 7 VID 1
# CUSTOMER SEGMENTATION WITH K-MEANS
# ============================================================

print("=" * 60)
print("WEEK 7 VID 1 - CUSTOMER SEGMENTATION")
print("=" * 60)


# ============================================================
# 1. LOAD CUSTOMER DATA
# ============================================================

df = pd.read_csv("customer_data.csv")

print("\nDataset loaded successfully.")
print(f"Number of customers: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SELECT FEATURES
# ============================================================

# These are the numerical features available in our dataset.

features = [
    "total_spend",
    "num_orders",
    "avg_order_value",
    "days_since_last_order",
    "num_support_tickets"
]

X = df[features].copy()

print("\nFeatures used for clustering:")

for feature in features:
    print(f"- {feature}")


# ============================================================
# 3. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled using StandardScaler.")


# ============================================================
# 4. ELBOW METHOD
# ============================================================

print("\n" + "=" * 60)
print("ELBOW METHOD")
print("=" * 60)

inertias = []

for k in range(1, 11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)

    inertias.append(
        km.inertia_
    )

    print(
        f"K = {k}, "
        f"Inertia = {km.inertia_:.2f}"
    )


# Plot elbow curve
plt.figure(figsize=(9, 4))

plt.plot(
    range(1, 11),
    inertias,
    "o-"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Choose Optimal K")
plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "elbow_plot.png",
    dpi=300
)

plt.show()

print(
    "\nElbow plot saved as 'elbow_plot.png'"
)


# ============================================================
# 5. CHOOSE OPTIMAL K
# ============================================================

# Based on the elbow observed in our dataset,
# we use 4 clusters.

OPTIMAL_K = 4

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

print(
    f"Optimal K selected: {OPTIMAL_K}"
)


# ============================================================
# 6. RUN K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,
    n_init=10
)

df["segment"] = kmeans.fit_predict(
    X_scaled
)

print(
    "\nK-Means clustering completed."
)


# ============================================================
# 7. PCA FOR VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]


# ============================================================
# 8. PCA EXPLAINED VARIANCE
# ============================================================

pc1_variance = (
    pca.explained_variance_ratio_[0]
    * 100
)

pc2_variance = (
    pca.explained_variance_ratio_[1]
    * 100
)

total_variance = (
    pc1_variance + pc2_variance
)

print("\n" + "=" * 60)
print("PCA EXPLAINED VARIANCE")
print("=" * 60)

print(
    f"PC1: {pc1_variance:.1f}%"
)

print(
    f"PC2: {pc2_variance:.1f}%"
)

print(
    f"PC1 + PC2: {total_variance:.1f}%"
)

if total_variance >= 60:

    print(
        "PC1 and PC2 explain at least "
        "60% of the variance."
    )

else:

    print(
        "PC1 and PC2 explain less than "
        "60% of the variance."
    )


# ============================================================
# 9. TRANSFORM CLUSTER CENTRES INTO PCA SPACE
# ============================================================

centers_pca = pca.transform(
    kmeans.cluster_centers_
)


# ============================================================
# 10. CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

for cluster in range(OPTIMAL_K):

    mask = (
        df["segment"] == cluster
    )

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Segment {cluster}",
        alpha=0.6,
        s=40
    )


# Plot cluster centres on the SAME figure
plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    marker="X",
    s=180,
    label="Centres"
)


plt.xlabel(
    f"PC1 ({pc1_variance:.1f}% variance)"
)

plt.ylabel(
    f"PC2 ({pc2_variance:.1f}% variance)"
)

plt.title(
    f"Customer Segments K={OPTIMAL_K}"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "cluster_visualization.png",
    dpi=300
)

plt.show()

print(
    "\nCluster visualization saved as "
    "'cluster_visualization.png'"
)


# ============================================================
# 11. CUSTOMER COUNT BY SEGMENT
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER COUNT BY SEGMENT")
print("=" * 60)

segment_counts = (
    df["segment"]
    .value_counts()
    .sort_index()
)

print(
    segment_counts
)


# ============================================================
# 12. PROFILE EACH SEGMENT
# ============================================================

print("\n" + "=" * 60)
print("SEGMENT PROFILES")
print("=" * 60)

profiles = (
    df.groupby("segment")[features]
    .mean()
    .round(2)
)

print(
    profiles
)


# Save profiles
profiles.to_csv(
    "cluster_profiles.csv"
)

print(
    "\nCluster profiles saved as "
    "'cluster_profiles.csv'"
)


# ============================================================
# 13. MEANINGFUL SEGMENT NAMES
# ============================================================

# These names are based on the actual profiles produced
# by our customer dataset.

segment_names = {
    0: "High Value Active Customers",
    1: "Regular Engaged Customers",
    2: "At Risk Customers",
    3: "Low Value Inactive Customers"
}

df["segment_name"] = (
    df["segment"]
    .map(segment_names)
)


# ============================================================
# 14. SEGMENT INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("SEGMENT INTERPRETATION")
print("=" * 60)


for segment in range(OPTIMAL_K):

    profile = profiles.loc[segment]

    count = segment_counts[segment]

    name = segment_names[segment]

    print("\n" + "-" * 55)

    print(
        f"Segment {segment}: {name}"
    )

    print(
        f"Customers: {count}"
    )

    print(
        f"Average Total Spend: "
        f"${profile['total_spend']:,.2f}"
    )

    print(
        f"Average Num Orders: "
        f"{profile['num_orders']:.1f}"
    )

    print(
        f"Average Order Value: "
        f"${profile['avg_order_value']:,.2f}"
    )

    print(
        f"Average Days Since Last Order: "
        f"{profile['days_since_last_order']:.1f}"
    )

    print(
        f"Average Support Tickets: "
        f"{profile['num_support_tickets']:.1f}"
    )


# ============================================================
# 15. SAVE CUSTOMER SEGMENTS
# ============================================================

df.to_csv(
    "customer_segments.csv",
    index=False
)

print(
    "\nCustomer segments saved as "
    "'customer_segments.csv'"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SEGMENT SUMMARY")
print("=" * 60)

for segment, name in segment_names.items():

    print(
        f"Segment {segment}: "
        f"{name} "
        f"({segment_counts[segment]} customers)"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("VID 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)
