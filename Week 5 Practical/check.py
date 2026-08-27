import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ==========================
# STEP 1: LOAD DATA
# ==========================

df = pd.read_csv("customer_churn.csv")

print("Customers:", len(df))
print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nChurn rate:")
print(df["Churn"].value_counts(normalize=True))

# ==========================
# STEP 2: EXPLORE AND CLEAN
# ==========================

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nDuplicate rows:", df.duplicated().sum())

# ==========================
# STEP 2: CLEAN DATA
# ==========================

# Remove duplicate rows
df = df.drop_duplicates()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing numerical values with the median
numeric_columns = df.select_dtypes(include=["number"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing categorical values with the mode
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nAfter cleaning:")
print("Rows:", len(df))
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# ==========================
# STEP 3: PREPARE FEATURES
# ==========================

# Remove customer ID because it does not help predict churn
df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical features into numerical features
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nFeature preparation completed.")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of features:", X.shape[1])

# ==========================
# STEP 4: IMPROVE PREPROCESSING
# AND COMPARE MODELS
# ==========================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        C=2
    )
}

# Store results
model_results = {}
trained_models = {}

print("\n========== IMPROVED MODEL PERFORMANCE ==========")

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    model_results[name] = accuracy
    trained_models[name] = pipeline

    print(f"{name}: {accuracy:.2%}")

# Find best model
best_model_name = max(
    model_results,
    key=model_results.get
)

best_accuracy = model_results[best_model_name]
best_model = trained_models[best_model_name]

print("\n========== BEST MODEL ==========")
print("Best model:", best_model_name)
print(f"Best accuracy: {best_accuracy:.2%}")

if best_accuracy >= 0.80:
    print("✅ Model achieved the 80% accuracy target.")
else:
    print("⚠️ Best model is still below the 80% target.")

    # ==========================
# STEP 5: TEST CATBOOST MODEL
# ==========================

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Use the original features with categorical columns
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# Identify categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

# Convert categorical column names to their column positions
categorical_indices = [
    X.columns.get_loc(column)
    for column in categorical_columns
]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create CatBoost model
catboost_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=False
)

# Train the model
catboost_model.fit(
    X_train,
    y_train,
    cat_features=categorical_indices
)

# Make predictions
catboost_predictions = catboost_model.predict(X_test)

# Calculate accuracy
catboost_accuracy = accuracy_score(
    y_test,
    catboost_predictions
)

print("\n========== CATBOOST RESULTS ==========")
print(f"CatBoost Accuracy: {catboost_accuracy:.2%}")

if catboost_accuracy >= 0.80:
    print("✅ CatBoost achieved the 80% accuracy target.")
else:
    print("⚠️ CatBoost is still below the 80% target.")

print("\n========== CATBOOST CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        catboost_predictions
    )
)

# ==========================
# STEP 6: TEST XGBOOST MODEL
# ==========================

from xgboost import XGBClassifier

# Use the prepared numerical features
X = pd.get_dummies(
    df.drop(columns=["Churn"]),
    drop_first=True
)

y = df["Churn"].map({"Yes": 1, "No": 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
xgb_predictions = xgb_model.predict(X_test)

# Accuracy
xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

print("\n========== XGBOOST RESULTS ==========")
print(f"XGBoost Accuracy: {xgb_accuracy:.2%}")

if xgb_accuracy >= 0.80:
    print("✅ XGBoost achieved the 80% accuracy target.")
else:
    print("⚠️ XGBoost is still below the 80% target.")

print("\n========== XGBOOST CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        xgb_predictions
    )
)

# ==========================
# STEP 7: DATASET INVESTIGATION
# ==========================

print("\n========== DATASET INVESTIGATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nTarget distribution:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))

print("\nData types:")
print(df.dtypes)

print("\nUnique values per column:")
print(df.nunique())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# ==========================
# STEP 8: FEATURE IMPORTANCE
# ==========================

print("\n========== FEATURE RELATIONSHIPS ==========")

# Convert categorical columns to numeric
analysis_df = pd.get_dummies(df, drop_first=True)

# Convert Churn to numeric
analysis_df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Calculate correlations with Churn
correlations = analysis_df.corr(numeric_only=True)["Churn"].drop("Churn")

# Sort by absolute correlation
correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

print("\nTop features related to Churn:")
print(correlations.head(15))

# ==========================================
# STEP 9: MODEL TUNING AND BEST MODEL SEARCH
# ==========================================

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import accuracy_score

print("\n========== MODEL TUNING ==========")

# Prepare data again for model tuning
model_df = df.copy()

# Remove customer ID if it still exists
if "customerID" in model_df.columns:
    model_df = model_df.drop(columns=["customerID"])

# Convert target to numbers
model_df["Churn"] = model_df["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical features to numbers
model_df = pd.get_dummies(model_df, drop_first=True)

# Separate features and target
X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", X.shape[1])

# Models and parameter settings
models = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "class_weight": [None, "balanced"]
        }
    ),

    "Extra Trees": (
        ExtraTreesClassifier(random_state=42),
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "class_weight": [None, "balanced"]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4]
        }
    ),

    "Hist Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=42),
        {
            "max_iter": [100, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_leaf_nodes": [15, 31]
        }
    )
}

results = []

for name, (model, parameters) in models.items():

    print(f"\nTesting {name}...")

    grid = GridSearchCV(
        model,
        parameters,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    predictions = grid.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Best Parameters": grid.best_params_
    })

    print(f"{name} accuracy: {accuracy:.2%}")
    print("Best parameters:", grid.best_params_)


# Find the best model
best_result = max(results, key=lambda x: x["Accuracy"])

print("\n========== BEST TUNED MODEL ==========")
print("Best model:", best_result["Model"])
print(f"Best accuracy: {best_result['Accuracy']:.2%}")
print("Best parameters:", best_result["Best Parameters"])

if best_result["Accuracy"] >= 0.80:
    print("✅ Model achieved the 80% accuracy target.")
else:
    print("⚠️ Best model is still below the 80% accuracy target.")

    # ==========================================
# STEP 10: XGBOOST MODEL SEARCH
# ==========================================

from xgboost import XGBClassifier

print("\n========== XGBOOST MODEL SEARCH ==========")

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb_parameters = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    xgb,
    xgb_parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("Searching for the best XGBoost settings...")

xgb_grid.fit(X_train, y_train)

xgb_predictions = xgb_grid.predict(X_test)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

print("\n========== XGBOOST RESULTS ==========")
print(f"XGBoost accuracy: {xgb_accuracy:.2%}")
print("Best parameters:", xgb_grid.best_params_)

if xgb_accuracy >= 0.80:
    print("✅ XGBoost achieved the 80% accuracy target.")
else:
    print("⚠️ XGBoost is still below the 80% target.")

    # ============================================================
# STEP 10: TUNE CATBOOST MODEL
# ============================================================

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

print("\n" + "=" * 55)
print("CATBOOST HYPERPARAMETER TUNING")
print("=" * 55)

# Load original dataset
cat_df = pd.read_csv("customer_churn.csv")

# Remove customer ID
cat_df = cat_df.drop(columns=["customerID"])

# Convert target
cat_df["Churn"] = cat_df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Separate features and target
X = cat_df.drop(columns=["Churn"])
y = cat_df["Churn"]

# Identify categorical columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Fill missing values
for column in categorical_features:
    X[column] = X[column].fillna(
        X[column].mode()[0]
    )

for column in X.select_dtypes(
    exclude=["object"]
).columns:
    X[column] = X[column].fillna(
        X[column].median()
    )

# Convert categorical columns to category dtype
for column in categorical_features:
    X[column] = X[column].astype(str)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Base CatBoost model
catboost_model = CatBoostClassifier(
    loss_function="Logloss",
    verbose=False,
    random_seed=42
)

# Hyperparameters to test
parameters = {
    "iterations": [300, 500, 800],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg": [3, 5, 10]
}

# Grid search
grid = GridSearchCV(
    estimator=catboost_model,
    param_grid=parameters,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("Searching for the best CatBoost parameters...")

grid.fit(
    X_train,
    y_train,
    cat_features=categorical_features
)

# Best model
best_catboost = grid.best_estimator_

# Predictions
predictions = best_catboost.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 55)
print("TUNED CATBOOST RESULTS")
print("=" * 55)

print(f"Best accuracy: {accuracy:.2%}")

print("\nBest parameters:")
print(grid.best_params_)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

if accuracy >= 0.80:
    print("\n✅ CatBoost achieved the 80% accuracy target!")
else:
    print("\n⚠️ CatBoost is still below 80%.")

    # ============================================================
# STEP 10: TUNE CATBOOST MODEL
# ============================================================

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

print("\n" + "=" * 55)
print("CATBOOST HYPERPARAMETER TUNING")
print("=" * 55)

# Load original dataset
cat_df = pd.read_csv("customer_churn.csv")

# Remove customer ID
cat_df = cat_df.drop(columns=["customerID"])

# Convert target
cat_df["Churn"] = cat_df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Separate features and target
X = cat_df.drop(columns=["Churn"])
y = cat_df["Churn"]

# Identify categorical columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Fill missing values
for column in categorical_features:
    X[column] = X[column].fillna(
        X[column].mode()[0]
    )

for column in X.select_dtypes(
    exclude=["object"]
).columns:
    X[column] = X[column].fillna(
        X[column].median()
    )

# Convert categorical columns to category dtype
for column in categorical_features:
    X[column] = X[column].astype(str)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Base CatBoost model
catboost_model = CatBoostClassifier(
    loss_function="Logloss",
    verbose=False,
    random_seed=42
)

# Hyperparameters to test
parameters = {
    "iterations": [300, 500, 800],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg": [3, 5, 10]
}

# Grid search
grid = GridSearchCV(
    estimator=catboost_model,
    param_grid=parameters,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("Searching for the best CatBoost parameters...")

grid.fit(
    X_train,
    y_train,
    cat_features=categorical_features
)

# Best model
best_catboost = grid.best_estimator_

# Predictions
predictions = best_catboost.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 55)
print("TUNED CATBOOST RESULTS")
print("=" * 55)

print(f"Best accuracy: {accuracy:.2%}")

print("\nBest parameters:")
print(grid.best_params_)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

if accuracy >= 0.80:
    print("\n✅ CatBoost achieved the 80% accuracy target!")
else:
    print("\n⚠️ CatBoost is still below 80%.")