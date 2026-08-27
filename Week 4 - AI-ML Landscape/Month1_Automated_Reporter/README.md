# 📊 Customer Insights Automated Reporter

## Overview

This project is an automated customer insights reporting system built with Python.

It combines customer sentiment data and customer review data, cleans and analyzes the datasets, generates visualizations, creates a professional Excel report, and automatically sends the completed report by email.

The project demonstrates a complete data automation workflow:

**Load → Clean → Analyze → Visualize → Report → Email**

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Analyze customer sentiment and behavior
- Examine customer reviews and ratings
- Identify important customer insights
- Generate clear and colorful data visualizations
- Produce a structured Excel report
- Automate email delivery of the completed report
- Maintain logs of successful pipeline executions

---

## ✨ Features

- Load customer sentiment data from CSV
- Load customer review data from CSV
- Clean and prepare datasets
- Analyze customer sentiment
- Analyze customer reviews and ratings
- Identify top customer regions and review locations
- Calculate response time and issue resolution metrics
- Generate colorful charts
- Create a multi-sheet Excel report
- Automatically email the generated report
- Record pipeline activity in log files
- Protect email credentials using environment variables

---

## 📁 Project Structure

```text
Customer_Insights/
│
├── data/
│   ├── Customer_Sentiment.csv
│   ├── reviews_data.csv
│   └── README.md
│
├── output/
│   ├── Customer_Insights_Report.xlsx
│   ├── sentiment_distribution.png
│   ├── rating_distribution.png
│   ├── customers_by_region.png
│   ├── issue_resolution.png
│   └── response_time_distribution.png
│
├── log/
│   └── customer_insights.log
│
├── src/
│   ├── analyzer.py
│   ├── data_cleaner.py
│   ├── data_loader.py
│   ├── email_sender.py
│   ├── report_generator.py
│   └── visualizer.py
│
├── test/
│
├── .env
├── .gitignore
├── config.py
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

---

## 🔄 Automation Workflow

The project follows this automated workflow:

```text
Customer Sentiment CSV
        │
        ▼
    Data Loading
        │
        ▼
    Data Cleaning
        │
        ▼
      Analysis
        │
        ├──────────────► Customer Insights
        │
        └──────────────► Review Insights
        │
        ▼
   Visualization
        │
        ▼
   Excel Report
        │
        ▼
   Email Delivery
        │
        ▼
       Log
```

---

## 📊 Analysis Performed

### Customer Insights

The system calculates:

- Total customers
- Average response time
- Sentiment distribution
- Top region
- Issue resolution rate

### Review Insights

The system calculates:

- Total reviews
- Average rating
- Rating distribution
- Top review location

---

## 🎨 Visualizations

The generated report includes colorful visualizations for:

- Customer sentiment distribution
- Customer rating distribution
- Customers by region
- Issue resolution rate
- Customer response time distribution

---

## 📑 Excel Report

The automated Excel report contains:

### Executive Summary

Provides the main customer and review metrics.

### Customer Data

Contains the cleaned customer sentiment dataset.

### Reviews Data

Contains the cleaned review dataset.

### Charts

Contains the generated customer insight visualizations.

The final report is saved as:

```text
output/Customer_Insights_Report.xlsx
```

---

## 📧 Email Automation

After the Excel report is generated, the system automatically sends it through Gmail.

The email contains the generated Customer Insights Excel report as an attachment.

Email credentials are stored using environment variables rather than being hardcoded in the Python source code.

---

## 🔐 Security

Sensitive credentials are not stored directly in the source code.

The project uses a `.env` file for email credentials.

The `.env` file is excluded from Git using `.gitignore`.

Never commit passwords, API keys, or other sensitive credentials to a public repository.

---

## 📝 Logging

The project records important pipeline events in:

```text
log/customer_insights.log
```

The log records events such as:

- Pipeline start
- Analysis completion
- Excel report creation
- Email delivery
- Pipeline completion

This makes it easier to monitor and troubleshoot automated runs.

---

## 💻 Requirements

- Python 3.x
- pandas
- matplotlib
- openpyxl
- python-dotenv

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

From the project directory, run:

```bash
python main.py
```

The complete automated workflow will then execute:

```text
Load
↓
Clean
↓
Analyze
↓
Visualize
↓
Generate Excel Report
↓
Send Email
↓
Create Log
```

---

## 📂 Data Sources

The project uses two datasets:

1. Customer sentiment dataset
2. Customer review dataset

The datasets are stored locally in the `data/` directory.

The `data/README.md` file provides additional information about the datasets and their sources.

---

## 🧪 Testing

The project includes a `test/` directory for automated tests of the data processing and analysis components.

The test suite currently includes:

```text
test/
└── test_pipeline.py
```

Run the tests from the project directory:

```bash
pytest
```

The current test suite contains 2 tests, and both tests pass successfully.

---

## 📌 Project Status

✅ Automated customer insights reporting pipeline implemented.

The system currently supports:

- Data loading
- Data cleaning
- Data analysis
- Visualization
- Excel report generation
- Automated email delivery
- Logging
- Environment-based credential management

✅ Automated customer insights reporting pipeline implemented.

The system currently supports:

- Data loading
- Data cleaning
- Data analysis
- Visualization
- Excel report generation
- Automated email delivery
- Logging
- Environment-based credential management

---

## 🎓 Project Context

This project was developed as part of the:

**NECA ICT Academy – AI/ML Landscape Program**

**Month 1 Project**

---

## 📜 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

## 👤 Author
## 🧪 Testing

**Team: [PYCODE TEAM]**

NECA ICT Academy – AI/ML Landscape