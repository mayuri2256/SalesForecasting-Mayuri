# SalesForecasting-Mayuri

# 📊 Sales Forecasting using Machine Learning

## 📌 Project Overview

This project analyzes historical retail sales data to identify business trends, forecast future sales, detect unusual sales patterns, segment products based on demand, and present insights through an interactive Streamlit dashboard.

The project was developed as part of a Data Science Internship and demonstrates end-to-end data analysis, machine learning, and dashboard deployment.

---

## 🎯 Objectives

- Analyze historical retail sales data.
- Perform data cleaning and exploratory data analysis (EDA).
- Study time series trends and seasonality.
- Build and compare multiple forecasting models.
- Detect anomalous sales patterns.
- Segment products based on demand using clustering.
- Develop an interactive Streamlit dashboard.
- Generate a business-friendly executive report.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Prophet
- XGBoost
- Scikit-learn
- Streamlit

---

## 📂 Project Structure

```
SalesForecasting_[Mayuri Sonawane]
│
├── analysis.ipynb
├── app.py
├── train.csv
├── requirements.txt
├── summary.pdf
├── README.md
└── charts/
```

---

## 📈 Tasks Completed

### ✅ Task 1
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Monthly & Yearly Sales Analysis

### ✅ Task 2
- Time Series Analysis
- Seasonal Decomposition
- ADF Stationarity Test

### ✅ Task 3
Built and compared three forecasting models:

- SARIMA
- Prophet
- XGBoost

### Model Performance

| Model | MAE | RMSE | MAPE |
|-------|------:|------:|------:|
| SARIMA | 18031.40 | 19009.18 | 18.97% |
| Prophet | 20250.79 | 22318.41 | 21.86% |
| **XGBoost** | **13915.32** | **18893.85** | **13.29%** |

**Best Model:** XGBoost

---

### ✅ Task 4

Category & Region Level Forecasting

Forecast completed for:

- Furniture
- Technology
- Office Supplies
- West Region
- East Region

---

### ✅ Task 5

Anomaly Detection

Methods Used:

- Isolation Forest
- Z-Score

Isolation Forest detected **7 anomalies**.

---

### ✅ Task 6

Product Demand Segmentation

Applied:

- K-Means Clustering
- PCA Visualization

Demand segments:

- High Volume, Stable Demand
- Growing Demand
- High Growth, High Volatility
- Low Volume, Stable Demand

---

### ✅ Task 7

Interactive Streamlit Dashboard

Features:

- Sales Overview Dashboard
- Forecast Explorer
- Anomaly Report
- Product Demand Segments

---

### ✅ Task 8

Executive Business Report

Prepared a business-oriented report including:

- Executive Summary
- Forecast Results
- Business Recommendations
- Product Segmentation
- Risks & Limitations

---

## 📊 Forecast Results

| Month | Forecast Sales |
|--------|---------------:|
| Month 1 | 51,037.70 |
| Month 2 | 30,091.78 |
| Month 3 | 61,376.88 |

---

## 💡 Key Insights

- Overall sales show a positive growth trend.
- XGBoost achieved the highest forecasting accuracy.
- Technology category and West region are expected to grow the fastest.
- Isolation Forest effectively detected unusual sales patterns.
- Demand segmentation can help optimize inventory planning.

---

## 🚀 How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```
## Project Links

Live Streamlit App:
https://salesforecasting-mayuri-vpz2xpggkzkhrzizkamyey.streamlit.app/
