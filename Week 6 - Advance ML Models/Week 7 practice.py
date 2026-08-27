import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def segment_customers(
    df,
    features,
    k_range=range(1, 11)
):
    # Select the required features
    X = df[features]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Store inertia values
    inertia_values = []

    # Test each K
    for k in k_range:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X_scaled)

        inertia_values.append(
            kmeans.inertia_
        )

    # Display elbow plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_range),
        inertia_values,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")

    plt.xticks(list(k_range))

    plt.grid(True)

    plt.show()

    return inertia_values