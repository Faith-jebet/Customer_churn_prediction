# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c8cbf)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Project-Week%201%20%7C%20ML%20Internship-success)

## Overview

Customer churn is a major challenge for telecommunications companies because losing existing customers can directly affect revenue and long-term business growth.

This project focuses on **understanding and framing a customer churn prediction problem** using the Telco Customer Churn dataset. The objective is to identify the factors associated with customer churn, understand the structure and quality of the data, perform exploratory data analysis, and prepare the dataset and machine learning strategy for future predictive modeling.

The project was completed as part of the **AnalystLab Africa Machine Learning Internship Programme – Week 1: Machine Learning Problem Framing & Data Understanding**. The assignment requires business understanding, dataset inspection, problem framing, exploratory analysis, and a machine learning proposal.

---

## Business Problem

ABC Communications Ltd wants to identify customers who are likely to leave its services before they actually churn.

The proposed machine learning solution should help the business:

* Identify customers at high risk of churn
* Understand the characteristics associated with customer churn
* Support proactive customer retention strategies
* Enable the business to prioritize customers who may require intervention
* Reduce potential revenue loss caused by customer attrition

### Machine Learning Problem

The problem is formulated as a **binary classification problem**.

The model will predict whether a customer is likely to:

* `Yes` → Churn
* `No` → Remain with the company

The assignment specifically requires identifying the target variable, input features, problem type, evaluation metrics, and suitable algorithms.

---

## Dataset

The project uses the **Telco Customer Churn Dataset**, which contains customer demographic information, account information, subscribed services, billing information, and churn status.

**Dataset:** Telco Customer Churn Dataset
**Source:** Kaggle – IBM Telco Customer Churn Dataset

The dataset contains:

* **7,043 customer records**
* **21 variables**
* Numerical and categorical features
* Customer service and subscription information
* Billing and payment information
* A binary `Churn` target variable

### Important Features

| Feature            | Description                               |
| ------------------ | ----------------------------------------- |
| `customerID`       | Unique customer identifier                |
| `gender`           | Customer gender                           |
| `SeniorCitizen`    | Whether the customer is a senior citizen  |
| `Partner`          | Whether the customer has a partner        |
| `Dependents`       | Whether the customer has dependents       |
| `tenure`           | Number of months the customer has stayed  |
| `PhoneService`     | Whether the customer has phone service    |
| `MultipleLines`    | Multiple phone lines subscription         |
| `InternetService`  | Type of internet service                  |
| `OnlineSecurity`   | Online security subscription              |
| `OnlineBackup`     | Online backup subscription                |
| `DeviceProtection` | Device protection subscription            |
| `TechSupport`      | Technical support subscription            |
| `StreamingTV`      | Streaming TV subscription                 |
| `StreamingMovies`  | Streaming movies subscription             |
| `Contract`         | Customer contract type                    |
| `PaperlessBilling` | Paperless billing status                  |
| `PaymentMethod`    | Customer payment method                   |
| `MonthlyCharges`   | Monthly amount charged                    |
| `TotalCharges`     | Total amount charged                      |
| `Churn`            | Target variable indicating customer churn |

---

## Project Objectives

The project aims to:

1. Understand customer churn from a business perspective
2. Inspect and understand the dataset
3. Identify the target variable and input features
4. Determine whether the problem is classification or regression
5. Identify data quality issues
6. Perform exploratory data analysis
7. Investigate relationships between customer characteristics and churn
8. Recommend appropriate machine learning algorithms
9. Identify suitable evaluation metrics
10. Define preprocessing requirements for future model development

These objectives align with the Week 1 assignment requirements provided by AnalystLab Africa.

---

## Project Workflow

```text
Business Understanding
        ↓
Dataset Collection
        ↓
Dataset Inspection
        ↓
Data Quality Assessment
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Problem Framing
        ↓
Feature & Target Identification
        ↓
Model Planning
        ↓
Future Model Development
```

---

## Data Inspection & Cleaning

The dataset was inspected to understand its structure, data types, missing values, and duplicate records.

### Dataset Structure

The dataset contains **7,043 rows and 21 columns**.

The variables include:

* Numerical features such as `tenure`, `MonthlyCharges`, and `SeniorCitizen`
* Categorical features such as `Contract`, `InternetService`, and `PaymentMethod`
* The target variable `Churn`

### Missing Values

Initial inspection did not identify conventional null values.

However, further inspection revealed that the `TotalCharges` column contained **blank string values** that were not initially recognized as standard missing values.

These values were converted to numeric format using `pd.to_numeric()` and the resulting missing values were handled by replacing them with `0`.

### Duplicate Records

The dataset was also checked for duplicate records.

Results:

* Fully duplicated rows: **0**
* Duplicated `customerID` values: **0**

This indicates that each customer record is uniquely identified in the dataset.

---

## Target Variable

The target variable is:

```text
Churn
```

It contains two possible outcomes:

```text
Yes
No
```

### Target Distribution

| Churn     | Customers | Percentage |
| --------- | --------: | ---------: |
| No        |     5,174 |     73.46% |
| Yes       |     1,869 |     26.54% |
| **Total** | **7,043** |   **100%** |

Approximately **26.54% of customers in the dataset have churned**, while **73.46% have remained**.

This indicates a degree of **class imbalance**, which should be considered during future model development and evaluation.

---

## Exploratory Data Analysis

The exploratory analysis was designed to investigate relationships between customer characteristics and churn.

The assignment requires at least:

* 3 bar charts
* 2 histograms
* 1 correlation heatmap

These visualizations were implemented in the notebook.

### 1. Customer Churn by Contract Type

The analysis compares churn across:

* Month-to-month
* One year
* Two year

This helps investigate whether contract duration is associated with customer retention.

### 2. Customer Churn by Internet Service

Customer churn was compared across different internet service types.

This provides insight into whether the type of internet service is associated with different churn patterns.

### 3. Customer Churn by Payment Method

The analysis compares churn across customer payment methods.

This can help identify payment-related patterns that may be associated with customer attrition.

### 4. Customer Tenure Distribution

A histogram was used to examine how customer tenure is distributed and how tenure relates to churn.

### 5. Monthly Charges Distribution

A histogram was used to analyze the distribution of monthly customer charges and compare it with churn status.

### 6. Correlation Analysis

A correlation heatmap was created using selected numerical variables:

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `ChurnFlag`

The `Churn` variable was converted into a binary `ChurnFlag`:

```text
Yes → 1
No  → 0
```

This allows numerical correlation analysis between customer attributes and churn.

---

## Key Exploratory Insights

The analysis provides several useful observations for future modeling:

### Customer churn is not evenly distributed

With approximately 26.54% of customers classified as churned, the target variable is imbalanced.

Therefore, model evaluation should not rely solely on accuracy.

### Contract type is an important candidate feature

Customer contract type is investigated because customers on different contract structures may demonstrate different retention behavior.

### Tenure may provide useful predictive information

Customer tenure represents how long a customer has remained with the company and can potentially provide useful information when predicting churn.

### Monthly charges may be relevant

Monthly charges are another important numerical feature that can be investigated as a potential indicator of customer churn.

> These observations are exploratory findings and should be validated statistically and through machine learning models before being treated as definitive causal relationships.

---

## Machine Learning Approach

Since the target variable contains two classes (`Yes` and `No`), the problem is a **binary classification task**.

### Candidate Algorithms

The following algorithms are suitable candidates for future model development:

#### Logistic Regression

A strong baseline model for binary classification.

Advantages:

* Easy to interpret
* Fast to train
* Provides probability estimates
* Useful as a baseline model

#### Decision Tree

Useful for understanding non-linear relationships between customer characteristics and churn.

Advantages:

* Easy to interpret
* Handles non-linear relationships
* Can capture feature interactions

#### Random Forest

An ensemble-based classification algorithm that combines multiple decision trees.

Advantages:

* Handles non-linear relationships
* Generally robust to noise
* Can provide feature importance
* Suitable for mixed feature types after preprocessing

#### Gradient Boosting

A powerful ensemble approach that can capture complex patterns in structured/tabular data.

It can be considered for improved predictive performance after establishing baseline models.

---

## Evaluation Metrics

Because the churn dataset is imbalanced, several evaluation metrics should be considered.

### Accuracy

Measures the percentage of predictions that are correct.

However, accuracy alone may be misleading when the classes are imbalanced.

### Precision

Measures how many customers predicted as churners actually churned.

### Recall

Measures how many actual churners were correctly identified.

For a customer retention system, **recall is particularly important** because failing to identify a customer who is about to churn may result in a missed retention opportunity.

### F1-Score

The F1-score balances precision and recall and can provide a more informative measure when dealing with imbalanced classes.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between churned and non-churned customers across different classification thresholds.

### Recommended Evaluation Strategy

The future model evaluation should consider:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
```

Rather than relying on accuracy alone.

---

## Preprocessing Strategy

Before training machine learning models, the following preprocessing steps will be required:

### 1. Remove Customer Identifier

`customerID` is a unique identifier and should generally not be used as a predictive feature.

### 2. Encode Categorical Variables

Categorical variables such as:

* `Contract`
* `InternetService`
* `PaymentMethod`
* `Partner`
* `Dependents`

will need to be converted into numerical representations.

Possible approaches include:

* One-hot encoding
* Binary encoding where appropriate

### 3. Scale Numerical Features

Numerical variables such as:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

may require scaling, particularly for algorithms such as Logistic Regression.

### 4. Encode Target Variable

The target can be converted to a binary representation:

```text
No  → 0
Yes → 1
```

### 5. Train-Test Split

The dataset should be divided into training and testing sets.

A stratified split should be considered to preserve the churn class distribution across both sets.

---

## Project Structure

```text
customer-churn-prediction/
│
├── Customer_Churn_Prediction_Project.ipynb
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── visualizations/
│   ├── chart1_churn_by_contract.png
│   ├── chart2_churn_by_internet.png
│   ├── chart3_churn_by_payment.png
│   ├── hist1_tenure.png
│   ├── hist2_monthlycharges.png
│   └── heatmap_correlation.png
│
├── README.md
│
└── requirements.txt
```

---

## Technologies & Tools

| Tool             | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Python           | Data analysis and machine learning        |
| Pandas           | Data manipulation and analysis            |
| NumPy            | Numerical computation                     |
| Matplotlib       | Data visualization                        |
| Seaborn          | Statistical visualization                 |
| Jupyter Notebook | Interactive analysis                      |
| Git & GitHub     | Version control and project documentation |

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

### 2. Navigate to the project

```bash
cd customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn jupyter openpyxl
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Customer_Churn_Prediction_Project.ipynb
```

and run the cells sequentially.

---

## Current Project Status

### Completed

* [x] Business problem identification
* [x] Dataset loading
* [x] Dataset inspection
* [x] Data type inspection
* [x] Missing-value investigation
* [x] Hidden blank-value detection
* [x] `TotalCharges` cleaning
* [x] Duplicate record checks
* [x] Descriptive statistics
* [x] Target variable analysis
* [x] Three bar charts
* [x] Two histograms
* [x] Correlation heatmap
* [x] Classification problem framing
* [x] Initial algorithm recommendations
* [x] Evaluation metric planning
* [x] Preprocessing strategy

### Future Work

* [ ] Feature engineering
* [ ] Train-test split
* [ ] Categorical encoding
* [ ] Feature scaling
* [ ] Model training
* [ ] Hyperparameter tuning
* [ ] Cross-validation
* [ ] Model comparison
* [ ] Feature importance analysis
* [ ] Confusion matrix analysis
* [ ] ROC-AUC evaluation
* [ ] Model selection
* [ ] Model deployment

---

## Expected Business Impact

A successful churn prediction model could help a telecommunications company:

* Identify customers at risk of leaving
* Prioritize retention campaigns
* Improve customer retention
* Reduce customer acquisition replacement costs
* Better understand customer behavior
* Make data-driven customer engagement decisions

The ultimate goal is not simply to predict churn, but to provide actionable information that can support **proactive customer retention**.

---

## Learning Outcomes

Through this project, I strengthened my understanding of:

* Machine learning problem formulation
* Binary classification
* Data quality assessment
* Data cleaning with Pandas
* Exploratory data analysis
* Categorical and numerical features
* Target variable analysis
* Data visualization
* Class imbalance
* Machine learning evaluation metrics
* Preprocessing strategies
* Translating business problems into machine learning problems

---

## Assignment Context

This project was completed as part of the **AnalystLab Africa Machine Learning Internship Programme – Week 1**.

The official Week 1 assignment focuses on:

> Machine Learning Problem Framing & Data Understanding

The required deliverables include a Business Understanding Report, Dataset Inspection Report, Jupyter Notebook, Machine Learning Proposal, GitHub repository, and professional social media documentation.

---

## Author

**Faith Jebet**

Information Technology Graduate | Aspiring Machine Learning Engineer | Data Analyst | Software Developer

### Areas of Interest

* Machine Learning
* Data Analytics
* Artificial Intelligence
* Software Engineering
* Data Visualization

---

## Acknowledgements

* **AnalystLab Africa** – Machine Learning Internship Programme
* **Kaggle / IBM Telco Customer Churn Dataset** – Dataset source
* **Python Data Science Community** – Open-source tools and libraries used in this project

---

## License

This project is intended for **educational and portfolio purposes**.

Dataset licensing and usage rights remain subject to the original dataset provider's terms.
