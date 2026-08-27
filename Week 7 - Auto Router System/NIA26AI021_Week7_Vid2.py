import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# WEEK 7 VID 2
# ANOMALY DETECTION WITH ISOLATION FOREST
# ============================================================

print("=" * 60)
print("WEEK 7 VID 2 - ANOMALY DETECTION")
print("=" * 60)


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv("transaction_data.csv")

print("\nDataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. SELECT NUMERICAL FEATURES
# ============================================================

numerical_cols = [
    "amount",
    "frequency",
    "time_of_day",
    "account_age"
]

X = df[numerical_cols].copy()


# Fill missing numerical values with the median
X = X.fillna(X.median())


# ============================================================
# 3. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData preprocessed and scaled.")


# ============================================================
# 4. ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 60)
print("ANOMALY DETECTION")
print("=" * 60)

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X_scaled)

# Isolation Forest:
#  1  = normal
# -1  = anomaly

df["is_anomaly"] = predictions == -1

n_anomalies = df["is_anomaly"].sum()

n_normal = len(df) - n_anomalies

print("\nANOMALY DETECTION RESULTS")
print("-" * 40)

print(
    f"Total transactions: {len(df)}"
)

print(
    f"Normal transactions: {n_normal}"
)

print(
    f"Anomalies detected: {n_anomalies}"
)

print(
    f"Anomaly rate: "
    f"{n_anomalies / len(df) * 100:.1f}%"
)


# ============================================================
# 5. COMPARE NORMAL AND ANOMALOUS PROFILES
# ============================================================

print("\n" + "=" * 60)
print("PROFILE COMPARISON")
print("=" * 60)


normal_avg = (
    df[df["is_anomaly"] == False][numerical_cols]
    .mean()
)

anomaly_avg = (
    df[df["is_anomaly"] == True][numerical_cols]
    .mean()
)


print("\nNORMAL TRANSACTIONS (average):")

for col in numerical_cols:

    print(
        f"{col} : "
        f"{normal_avg[col]:.2f}"
    )


print("\nANOMALOUS TRANSACTIONS (average):")

for col in numerical_cols:

    print(
        f"{col} : "
        f"{anomaly_avg[col]:.2f}"
    )


print("\nDIFFERENCE (anomaly minus normal):")

for col in numerical_cols:

    difference = (
        anomaly_avg[col]
        - normal_avg[col]
    )

    print(
        f"{col} : "
        f"{difference:.2f}"
    )


# Save profile comparison
profile_comparison = pd.DataFrame({
    "Normal": normal_avg,
    "Anomaly": anomaly_avg,
    "Difference": anomaly_avg - normal_avg
})

profile_comparison.to_csv(
    "anomaly_profile_comparison.csv"
)

print(
    "\nProfile comparison saved as "
    "'anomaly_profile_comparison.csv'"
)


# ============================================================
# 6. PCA DEEP DIVE
# ============================================================

print("\n" + "=" * 60)
print("PCA VARIANCE EXPLAINED")
print("=" * 60)

pca_full = PCA()

pca_full.fit(X_scaled)

cumulative = 0

for i, variance in enumerate(
    pca_full.explained_variance_ratio_,
    start=1
):

    cumulative += variance

    print(
        f"Component {i}: "
        f"{variance * 100:.1f}% "
        f"(Cumulative: "
        f"{cumulative * 100:.1f}%)"
    )


# ============================================================
# 7. PCA VARIANCE PLOTS
# ============================================================

components = range(
    1,
    len(
        pca_full.explained_variance_ratio_
    ) + 1
)

variance_percent = (
    pca_full.explained_variance_ratio_
    * 100
)

cumulative_percent = (
    pca_full.explained_variance_ratio_
    .cumsum()
    * 100
)


fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# Cumulative variance
ax1.plot(
    components,
    cumulative_percent,
    marker="o"
)

ax1.axhline(
    y=95,
    linestyle="--",
    label="95% threshold"
)

ax1.set_xlabel(
    "Number of Components"
)

ax1.set_ylabel(
    "Cumulative Explained Variance"
)

ax1.set_title(
    "How Many Components to Keep?"
)

ax1.legend()

ax1.grid(True)


# Variance by component
ax2.bar(
    components,
    variance_percent
)

ax2.set_xlabel(
    "Principal Component"
)

ax2.set_ylabel(
    "Explained Variance Ratio"
)

ax2.set_title(
    "Variance Explained by Each Component"
)

plt.tight_layout()

plt.savefig(
    "pca_variance.png",
    dpi=300
)

plt.show()

print(
    "\nPCA variance plot saved as "
    "'pca_variance.png'"
)


# ============================================================
# 8. VISUALIZE ANOMALIES WITH PCA
# ============================================================

pca_2d = PCA(
    n_components=2
)

X_pca = pca_2d.fit_transform(
    X_scaled
)


plt.figure(
    figsize=(16, 10)
)


normal_mask = (
    df["is_anomaly"] == False
)

anomaly_mask = (
    df["is_anomaly"] == True
)


# Normal transactions
plt.scatter(
    X_pca[normal_mask, 0],
    X_pca[normal_mask, 1],
    c="blue",
    label="Normal",
    alpha=0.4,
    s=20
)


# Anomalous transactions
plt.scatter(
    X_pca[anomaly_mask, 0],
    X_pca[anomaly_mask, 1],
    c="red",
    label="Anomaly",
    alpha=0.9,
    s=60,
    edgecolors="black"
)


pc1_variance = (
    pca_2d.explained_variance_ratio_[0]
    * 100
)

pc2_variance = (
    pca_2d.explained_variance_ratio_[1]
    * 100
)


plt.xlabel(
    f"Principal Component 1 "
    f"({pc1_variance:.1f}%)"
)

plt.ylabel(
    f"Principal Component 2 "
    f"({pc2_variance:.1f}%)"
)

plt.title(
    "PCA: Anomalies vs Normal Transactions",
    fontweight="bold"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "anomaly_pca_visualization.png",
    dpi=300
)

plt.show()

print(
    "\nAnomaly PCA visualization saved as "
    "'anomaly_pca_visualization.png'"
)


# ============================================================
# 9. CONTAMINATION EXPERIMENT
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENTING WITH CONTAMINATION VALUES")
print("=" * 60)

contamination_values = [
    0.02,
    0.05,
    0.10,
    0.15
]

contamination_results = []


for contamination in contamination_values:

    test_model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    test_predictions = (
        test_model.fit_predict(X_scaled)
    )

    anomaly_count = (
        test_predictions == -1
    ).sum()

    percentage = (
        anomaly_count
        / len(df)
        * 100
    )

    contamination_results.append({
        "Contamination": contamination,
        "Anomalies": anomaly_count,
        "Percentage": percentage
    })

    print(
        f"Contamination = {contamination:.2f} "
        f"-> {anomaly_count} anomalies "
        f"({percentage:.1f}%)"
    )


contamination_df = pd.DataFrame(
    contamination_results
)

contamination_df.to_csv(
    "contamination_comparison.csv",
    index=False
)

print(
    "\nContamination comparison saved as "
    "'contamination_comparison.csv'"
)



print("\n" + "=" * 60)
print("ANOMALY PROFILE BY CONTAMINATION LEVEL")
print("=" * 60)

profile_results = []

for contamination in contamination_values:

    test_model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    test_predictions = test_model.fit_predict(
        X_scaled
    )

    normal_data = X_scaled[test_predictions == 1]
    anomaly_data = X_scaled[test_predictions == -1]

    # Compare the average standardized profile
    normal_mean = normal_data.mean(axis=0)
    anomaly_mean = anomaly_data.mean(axis=0)

    average_difference = (
        abs(anomaly_mean - normal_mean).mean()
    )

    profile_results.append({
        "Contamination": contamination,
        "Anomalies": (test_predictions == -1).sum(),
        "Average Standardized Difference":
            average_difference
    })

    print(
        f"\nContamination = {contamination:.2f}"
    )

    print(
        f"Anomalies: "
        f"{(test_predictions == -1).sum()}"
    )

    print(
        f"Average standardized difference: "
        f"{average_difference:.2f}"
    )


profile_analysis = pd.DataFrame(
    profile_results
)

print("\n" + "-" * 60)
print("CONTAMINATION PROFILE SUMMARY")
print("-" * 60)

print(
    profile_analysis.round(2).to_string(
        index=False
    )
)

profile_analysis.to_csv(
    "contamination_profile_analysis.csv",
    index=False
)

print(
    "\nSaved as "
    "'contamination_profile_analysis.csv'"
)


# Identify the point where anomalies are least distinct
weakest = profile_analysis.loc[
    profile_analysis[
        "Average Standardized Difference"
    ].idxmin()
]

print(
    f"\nLowest anomaly separation occurs at "
    f"contamination = "
    f"{weakest['Contamination']:.2f}"
)

print(
    "At this level, the flagged transactions are "
    "least distinct from normal transactions."
)

# ============================================================
# 10. SAVE FINAL RESULTS
# ============================================================

df.to_csv(
    "anomaly_detection_results.csv",
    index=False
)

print(
    "\nFinal anomaly results saved as "
    "'anomaly_detection_results.csv'"
)


# ============================================================
# 11. BUSINESS QUESTION
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS QUESTION")
print("=" * 60)

print("""
In a real fraud detection system, I would rather have
some false positives than false negatives.

Missing an actual fraudulent transaction can lead to
financial loss and security risks. False positives can
temporarily inconvenience legitimate customers, but they
can be reviewed before taking further action.

The detection threshold should therefore be chosen based
on the relative business cost of false positives and
false negatives.
""")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("VID 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)