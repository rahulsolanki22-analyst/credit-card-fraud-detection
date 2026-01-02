**Credit Card Fraud Detection using Machine Learning**


Overview

This project is an end-to-end Credit Card Fraud Detection system built using machine learning.
The goal is to identify whether a transaction is fraudulent or normal using historical transaction data.

Fraud detection is a challenging problem because fraud cases are extremely rare, so accuracy alone is not reliable. This project focuses on the right evaluation metrics and a realistic ML workflow.

Problem Statement

Credit card fraud detection is a highly imbalanced classification problem, where fraudulent transactions make up less than 0.2% of the data.

The objective is to:

Detect fraudulent transactions

Catch as many frauds as possible

Reduce false alarms

Build a model that works well in real-world conditions

Dataset

Dataset: European Credit Card Transactions

Total transactions: 284,807

Fraud cases: 492

Target column: Class

0 → Normal transaction

1 → Fraudulent transaction

Features:

V1 to V28: PCA-transformed features (anonymized)

Time: Time since first transaction

Amount: Transaction amount

Because features are anonymized using PCA, they are not directly interpretable.

Project Structure

fraud_detection/
│
├── app/ # Streamlit app
│ └── app.py
│
├── data/
│ ├── raw/ # Original dataset (not uploaded)
│ └── processed/ # Cleaned data (not uploaded)
│
├── notebooks/ # Step-by-step notebooks
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_evaluation.ipynb
│
├── src/ # Reusable ML code
│
├── models/ # Trained models (not uploaded)
│
├── requirements.txt
└── README.md

Approach
1. Exploratory Data Analysis (EDA)

Checked data shape and columns

Verified there are no missing values

Analyzed class imbalance

Studied transaction amount distribution

Key finding:

Dataset is extremely imbalanced, so accuracy is misleading.

2. Data Preprocessing

Scaled Time and Amount using StandardScaler

Split data into train and test sets

Applied SMOTE only on training data to handle imbalance

Saved cleaned training data

3. Model Training

Trained and compared multiple models:

Logistic Regression (baseline)

Random Forest (final selected model)

XGBoost (advanced model)

Models were compared using:

Precision

Recall

F1-score

ROC-AUC

4. Model Evaluation

Accuracy was not used for model selection.

Key observations:

Logistic Regression had high recall but too many false positives

XGBoost had strong recall and ROC-AUC but was aggressive

Random Forest provided the best balance between fraud detection and false alarms

Final model selected: Random Forest

Streamlit Web App

A Streamlit application was built to demonstrate the model.

App Features

Input transaction details

Demo Mode (to showcase fraud detection clearly)

Load Sample Fraud Transaction button

Clear prediction output with confidence/probability

Demo Mode Explanation

Because the dataset uses PCA-transformed features, real-world inputs are not intuitive.
To solve this, the app includes a Demo Mode that consistently shows fraud detection behavior for presentation purposes.

When Demo Mode is OFF, the app uses real model predictions.

Technologies Used

Python
Pandas, NumPy
Scikit-learn
Imbalanced-learn (SMOTE)
XGBoost
Joblib
Streamlit
Matplotlib, Seaborn

How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Run the Streamlit app
streamlit run app/app.py

Final Conclusion

This project demonstrates a complete machine learning workflow for fraud detection on imbalanced data.
It highlights the importance of proper preprocessing, correct evaluation metrics, and practical deployment considerations.

The final Random Forest model provides a strong balance between detecting fraud and minimizing false alerts, making it suitable for real-world use cases.
