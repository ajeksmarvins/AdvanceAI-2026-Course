# Intelligent Decision Automator — Fraud Detection System

**Author: PyCode Team**

## Project Overview

This project is a machine learning-based fraud detection system developed as a Month 2 project.

The system analyzes transaction information and predicts whether a transaction is likely to be fraudulent. It also assigns a risk level and recommends an action based on the fraud probability.

## Project Objective

The objective is to build an intelligent system that can identify potentially fraudulent transactions and support faster decision-making.

The system produces:

- Fraud probability
- Prediction
- Confidence score
- Risk level
- Recommended action

## Decision Thresholds

| Fraud Probability | Risk Level | Action |
|---|---|---|
| 80% or higher | HIGH | AUTO-ACT |
| 50% – 79% | MEDIUM | HUMAN REVIEW |
| Below 50% | LOW | NO ACTION |

## Dataset

The project uses a fraud transaction dataset stored in:

data/fraud_transactions.csv

The target variable is:

is_fraud

Where:

- 0 = Legitimate transaction
- 1 = Fraudulent transaction

### Features

The model uses the following features:

- transaction_amount
- account_age_days
- transaction_hour
- previous_transactions
- device_type
- location_match
- amount_vs_average
- is_new_device

## Machine Learning Models

Three machine learning models are trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Recall and F1 Score receive particular attention because fraud detection is an imbalanced classification problem.

## Data Preprocessing

The preprocessing pipeline handles numerical and categorical features.

### Numerical Features

- Missing values are replaced using the median.
- Features are scaled using StandardScaler.

### Categorical Features

- Missing values are replaced using the most frequent value.
- Categories are converted using OneHotEncoder.

The preprocessing is included inside the machine learning pipeline to reduce the risk of data leakage.

## Class Imbalance

Fraudulent transactions are usually less common than legitimate transactions.

The project handles class imbalance using:

- class_weight="balanced" for Logistic Regression and Random Forest.
- scale_pos_weight for XGBoost.

## Model Validation

The dataset is divided into:

- 80% training data
- 20% testing data

Stratified 5-Fold Cross-Validation is also used to evaluate model performance while maintaining the class distribution.

## Hyperparameter Tuning

After comparing the models, the best-performing model is selected for hyperparameter tuning.

GridSearchCV is used to test different parameter combinations and select the best configuration based on F1 Score.

## Final Model Evaluation

The final model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

A classification report and confusion matrix are also generated.

The confusion matrix shows:

- True Negatives
- False Positives
- False Negatives
- True Positives

## Business Impact

The project includes an illustrative business-impact analysis.

It estimates:

- Value of correctly detected fraud
- Cost of missed fraud
- Cost of false alerts
- Estimated net impact

The financial values used are academic assumptions and can be replaced with real business costs.

## Automated Decision System

The system converts fraud probabilities into automated decisions.

### AUTO-ACT

Transactions with a fraud probability of 80% or higher are classified as HIGH risk and receive:

AUTO-ACT

### HUMAN REVIEW

Transactions with a fraud probability between 50% and 79% receive:

HUMAN REVIEW

### NO ACTION

Transactions with a fraud probability below 50% receive:

NO ACTION

## Prediction System

New transactions can be supplied through:

data/new_transactions.csv

The system generates:

- Fraud probability
- Confidence
- Prediction
- Risk level
- Recommended action
- Prediction timestamp

If a new transaction file does not exist, the system automatically creates a demonstration file using sample test records.

## SQLite Prediction Logging

Prediction results are stored in:

outputs/fraud_predictions.db

The database records:

- Prediction timestamp
- Fraud probability
- Confidence
- Prediction
- Risk level
- Action

This provides a simple record of automated decisions.

## Project Structure

MONTH 2 PROJECT/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── fraud_transactions.csv
│   └── new_transactions.csv
│
├── models/
│   ├── fraud_detection_model.pkl
│   ├── fraud_preprocessor.pkl
│   └── feature_schema.pkl
│
└── outputs/
    ├── target_distribution.png
    ├── confusion_matrix.png
    ├── model_comparison.csv
    ├── cross_validation_results.csv
    ├── final_model_metrics.csv
    ├── business_impact.csv
    ├── fraud_predictions.csv
    ├── fraud_predictions.db
    └── automation.log

## Installation

Install the required Python packages with:

pip install -r requirements.txt

## Running the Project

From the project root folder, run:

python main.py

The program will:

1. Load the dataset
2. Perform exploratory data analysis
3. Analyze fraud distribution
4. Preprocess the data
5. Train three machine learning models
6. Compare model performance
7. Perform cross-validation
8. Tune the selected model
9. Evaluate the final model
10. Generate a confusion matrix
11. Perform business-impact analysis
12. Save the trained model
13. Generate predictions for new transactions
14. Assign risk levels and actions
15. Log predictions in SQLite

## Output Files

Important outputs are saved in the outputs folder.

### model_comparison.csv

Contains the performance comparison of the machine learning models.

### cross_validation_results.csv

Contains the results of stratified 5-fold cross-validation.

### final_model_metrics.csv

Contains the final model evaluation metrics.

### confusion_matrix.png

Visualizes the final model's classification results.

### fraud_predictions.csv

Contains predictions, fraud probabilities, confidence scores, risk levels, and recommended actions.

### business_impact.csv

Contains the illustrative business-impact calculations.

### fraud_predictions.db

SQLite database containing prediction records.

### automation.log

Contains application logs generated while the system runs.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- SQLite

## Key Learning Outcomes

This project demonstrates:

- Classification
- Data preprocessing
- Handling imbalanced datasets
- Machine learning pipelines
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Probability-based prediction
- Decision automation
- Model persistence
- Database logging

## Conclusion

The Intelligent Decision Automator demonstrates how machine learning can be used to detect potentially fraudulent transactions and convert predictions into actionable decisions.

Rather than only predicting whether a transaction is fraudulent, the system provides a fraud probability, risk level, confidence score, and recommended action.

The project can be further extended with real-time transaction monitoring, API integration, dashboards, notifications, and production deployment.