from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def handle_imbalance_and_tune(X_train, y_train):

    # Check class distribution
    class_counts = y_train.value_counts()

    majority = class_counts.max()
    minority = class_counts.min()

    imbalance_ratio = majority / minority

    print("Class distribution:")
    print(class_counts)

    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

    # Apply SMOTE
    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE:")
    print(y_resampled.value_counts())

    # Random Forest
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    # Train
    grid_search.fit(
        X_resampled,
        y_resampled
    )

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params__)

    print(
        f"\nBest F1 score: "
        f"{grid_search.best_score_:.4f}"
    )

    return best_model, grid_search.best_score_