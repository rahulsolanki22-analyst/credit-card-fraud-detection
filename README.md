# Credit Card Fraud Detection using Machine Learning

## Overview
This project is an end-to-end machine learning system for detecting fraudulent credit card transactions from highly imbalanced data.  
The focus is on using correct evaluation metrics, handling class imbalance properly, and building a realistic ML workflow with deployment.

---

## Problem Statement
Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions account for less than 0.2% of the data.

Goals:
- Detect fraudulent transactions
- Catch as many frauds as possible
- Reduce false alarms
- Build a model suitable for real-world conditions

---

## Dataset
- Dataset: European Credit Card Transactions
- Total transactions: 284,807
- Fraud cases: 492
- Target column: Class
  - 0 = Normal transaction
  - 1 = Fraudulent transaction

Features:
- V1 to V28: PCA-transformed and anonymized features
- Time: Time since first transaction
- Amount: Transaction amount

Note: Due to PCA anonymization, individual features are not directly interpretable.

---

## Project Structure
credit-card-fraud-detection/
- app/
  - app.py (Streamlit web application)
- data/
  - raw/ (dataset not uploaded)
  - processed/ (cleaned data not uploaded)
- notebooks/
  - 01_eda.ipynb
  - 02_preprocessing.ipynb
  - 03_model_training.ipynb
  - 04_evaluation.ipynb
- src/
- models/ (saved models not uploaded)
- requirements.txt
- README.md

---

## Approach

### Exploratory Data Analysis
- Checked dataset shape and columns
- Verified there are no missing values
- Analyzed severe class imbalance
- Studied transaction amount distribution

Key insight:
- Accuracy is misleading due to extreme class imbalance

---

### Data Preprocessing
- Scaled Time and Amount using StandardScaler
- Split data into training and test sets
- Applied SMOTE only on training data to avoid data leakage

---

### Model Training
Trained and compared the following models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Evaluation metrics used:
- Precision
- Recall
- F1-score
- ROC-AUC

---

### Model Evaluation and Selection
- Logistic Regression showed high recall but many false positives
- XGBoost achieved strong recall and ROC-AUC but was aggressive
- Random Forest provided the best balance between fraud detection and false alarms

Final selected model:
- Random Forest

---

## Streamlit Web Application
A Streamlit web app was built to demonstrate real-time fraud detection.

App features:
- Transaction input interface
- Prediction output with probability
- Demo Mode for presentation-friendly fraud detection

Demo Mode explanation:
Since the dataset uses PCA-transformed features, realistic manual inputs are not intuitive.  
Demo Mode allows consistent demonstration of fraud detection behavior during presentations.

---

## Technologies Used
- Python
- Pandas
- NumPy
- scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

---

## How to Run the Project
1. Install dependencies
   pip install -r requirements.txt

2. Run the Streamlit application
   streamlit run app/app.py

---

## Conclusion
This project demonstrates a complete machine learning workflow for fraud detection on highly imbalanced data.  
It highlights the importance of proper preprocessing, correct evaluation metrics, and practical deployment considerations.

The final Random Forest model provides a strong balance between detecting fraud and minimizing false alerts, making it suitable for real-world applications.
