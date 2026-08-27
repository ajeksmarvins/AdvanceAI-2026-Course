import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("customer_churn.csv")

print("=== DATASET ===")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# =========================================================
# 2. CLEAN DATA
# =========================================================

df = df.drop_duplicates()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing numeric values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove customer ID
df = df.drop(
    columns=["customerID"],
    errors="ignore"
)


# =========================================================
# 3. PREPARE FEATURES
# =========================================================

le = LabelEncoder()

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Separate features and target
X = df.drop(columns=["Churn"])

y = df["Churn"]

print("\n=== FEATURES ===")
print("Number of features:", X.shape[1])


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================================================
# 5. DECISION TREE
# =========================================================

decision_tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

tree_predictions = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

print("\n=== DECISION TREE RESULTS ===")
print(f"Accuracy: {tree_accuracy:.2%}")


# =========================================================
# 6. VISUALIZE DECISION TREE
# =========================================================

plt.figure(figsize=(24, 12))

plot_tree(
    decision_tree,
    feature_names=X.columns,
    class_names=["Stay", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Customer Churn")
plt.tight_layout()

plt.savefig(
    "decision_tree_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 7. TRACE DECISION PATHS
# =========================================================

print("\n=== DECISION PATH ANALYSIS ===")

sample_indices = [0, 1, 2]

node_indicator = decision_tree.decision_path(X_test)
leaf_ids = decision_tree.apply(X_test)

for sample_number, sample_index in enumerate(sample_indices, start=1):

    if sample_index >= len(X_test):
        continue

    sample = X_test.iloc[[sample_index]]

    prediction = decision_tree.predict(sample)[0]

    print("\n----------------------------------------")
    print(f"Sample {sample_number}")
    print("----------------------------------------")

    print("Actual result:",
          "Churn" if y_test.iloc[sample_index] == 1 else "Stay")

    print("Predicted result:",
          "Churn" if prediction == 1 else "Stay")

    print("Decision path:")

    node_index = node_indicator.indices[
        node_indicator.indptr[sample_index]:
        node_indicator.indptr[sample_index + 1]
    ]

    for node in node_index:

        if leaf_ids[sample_index] == node:
            continue

        feature = decision_tree.tree_.feature[node]
        threshold = decision_tree.tree_.threshold[node]

        feature_name = X.columns[feature]

        sample_value = sample.iloc[0][feature_name]

        if sample_value <= threshold:
            direction = "<="
        else:
            direction = ">"

        print(
            f"{feature_name} = {sample_value:.2f} "
            f"{direction} {threshold:.2f}"
        )

    print(
        "\nWhy the model made this decision: "
        "The sample followed the decision rules above "
        "until it reached a final leaf node, where the "
        "model assigned the predicted class."
    )


# =========================================================
# 8. RANDOM FOREST
# =========================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

random_forest.fit(
    X_train,
    y_train
)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Accuracy: {rf_accuracy:.2%}")


# =========================================================
# 9. FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
print(feature_importance)


# =========================================================
# 10. FEATURE IMPORTANCE VISUALIZATION
# =========================================================

plt.figure(figsize=(12, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 11. FAIR LENDING QUESTION
# =========================================================

print("\n=== FAIR LENDING QUESTION ===")

print(
    "For a loan approval system that must comply with "
    "fair lending regulations, I would prefer a Decision "
    "Tree over a Random Forest when transparency and "
    "explainability are the priority."
)

print(
    "A Decision Tree provides a clear sequence of rules "
    "that can be inspected and explained for individual "
    "loan decisions."
)

print(
    "A Random Forest may provide stronger predictive "
    "performance, but it combines many trees, making "
    "individual decisions more difficult to explain."
)

print(
    "For a regulated lending system, explainability and "
    "the ability to audit decisions are especially "
    "important."
)


# =========================================================
# 12. BONUS: TREE DEPTH ANALYSIS
# =========================================================

print("\n=== BONUS: TREE DEPTH ANALYSIS ===")

depths = [2, 4, 8, 16]

train_accuracies = []
test_accuracies = []

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(
        X_train,
        y_train
    )

    train_prediction = tree.predict(X_train)
    test_prediction = tree.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(
        f"Depth {depth}: "
        f"Training Accuracy = {train_accuracy:.2%}, "
        f"Testing Accuracy = {test_accuracy:.2%}"
    )


# =========================================================
# 13. PLOT ACCURACY AGAINST DEPTH
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    depths,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depths,
    test_accuracies,
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs Depth")

plt.xticks(depths)
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "decision_tree_depth_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 14. OVERFITTING ANALYSIS
# =========================================================

print("\n=== OVERFITTING ANALYSIS ===")

for i, depth in enumerate(depths):

    difference = (
        train_accuracies[i]
        - test_accuracies[i]
    )

    print(
        f"Depth {depth}: "
        f"Train-Test Accuracy Gap = "
        f"{difference:.2%}"
    )

print(
    "\nA large increase in training accuracy while "
    "testing accuracy stops improving or decreases "
    "is a sign that the tree is beginning to overfit."
)


# =========================================================
# 15. FINAL COMPARISON
# =========================================================

print("\n=== FINAL MODEL COMPARISON ===")

print(
    f"Decision Tree Accuracy: "
    f"{tree_accuracy:.2%}"
)

print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy:.2%}"
)

print("\n=== PRACTICE 3 COMPLETE ===")