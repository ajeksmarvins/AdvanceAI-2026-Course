# 📊 Customer Insights — Data Documentation

## Overview

This folder contains the datasets used by the Customer Insights Automated Reporter.

The project uses two complementary customer-feedback sources:

1. A customer sentiment dataset
2. A web-scraped customer reviews dataset

The datasets are loaded, cleaned, analyzed, and used to generate customer insights and an automated Excel report.

---

## Dataset 1 — Customer Sentiment

**File:** `Customer_Sentiment.csv`

### Description

This dataset contains structured customer feedback information used for customer sentiment and service-performance analysis.

### Main Fields

| Field                  | Description                             |
| ---------------------- | --------------------------------------- |
| `customer_id`          | Unique customer identifier              |
| `gender`               | Customer gender                         |
| `age_group`            | Customer age group                      |
| `region`               | Customer region                         |
| `product_category`     | Product category                        |
| `purchase_channel`     | Purchase channel                        |
| `platform`             | Customer platform                       |
| `customer_rating`      | Customer rating                         |
| `review_text`          | Customer feedback text                  |
| `sentiment`            | Customer sentiment classification       |
| `response_time_hours`  | Response time in hours                  |
| `issue_resolved`       | Whether the reported issue was resolved |
| `complaint_registered` | Whether a complaint was registered      |

### Project Usage

This dataset is used to calculate:

* Total customers
* Average response time
* Sentiment distribution
* Top customer region
* Issue resolution rate

### Source and License

The original source and license information for this dataset should be verified from the original dataset page before redistribution.

This project does not claim ownership of the original dataset.

---

## Dataset 2 — Web-Scraped Customer Reviews

**File:** `reviews_data.csv`

### Description

This dataset contains customer reviews and ratings for Starbucks.

The Kaggle dataset describes the data as web-scraped customer reviews and ratings collected from the ConsumerAffairs website. It includes review text, star ratings, locations, dates, and image links.

### Main Fields

| Field         | Description                             |
| ------------- | --------------------------------------- |
| `name`        | Reviewer name                           |
| `location`    | Reviewer location                       |
| `Date`        | Review date                             |
| `Rating`      | Customer rating from 1 to 5             |
| `Review`      | Customer review text                    |
| `Image_Links` | Links to review images, where available |

### Project Usage

This dataset is used to calculate:

* Total reviews
* Average rating
* Rating distribution
* Top review location
* Customer review insights

### Original Source

**Kaggle Dataset:** Starbucks Reviews Dataset

**Author:** Harshal H

**Kaggle:**
https://www.kaggle.com/datasets/harshalhonde/starbucks-reviews-dataset

The Kaggle dataset states that the underlying reviews were collected through web scraping from ConsumerAffairs.

### License

The Kaggle dataset is listed under:

**Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

The dataset should be used in accordance with its stated license and applicable source-site terms.

---

## Data Processing

The project does not modify the original source files directly.

During the automated pipeline:

1. The CSV files are loaded.
2. Duplicate records are removed.
3. Missing values are handled.
4. Text fields are cleaned.
5. Ratings are converted to numeric values.
6. Review dates are converted to date format.
7. Invalid review ratings are removed.
8. The cleaned datasets are analyzed.
9. The resulting insights are included in the automated Excel report.

---

## Data Quality

The automated cleaning process is designed to handle common data-quality issues such as:

* Duplicate records
* Missing values
* Invalid ratings
* Inconsistent text formatting
* Invalid dates
* Empty review fields

Cleaning results are reported in the terminal during execution.

---

## Important Note

The datasets are included for this educational project.

Users should verify the original dataset pages, licenses, attribution requirements, and applicable source-site terms before redistributing the datasets or using them for commercial purposes.

The project code is separately licensed under the license specified in the root `LICENSE` file.
