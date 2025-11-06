# Online Payment Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-orange?logo=streamlit)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

---

## **Project Overview**
Online payments are one of the most widely used transaction methods globally. However, the growth of online transactions has also led to an increase in payment fraud.  

The goal of this project is to **train machine learning models to detect fraudulent transactions** using historical transaction data. Users can **upload their own transaction CSV files** through a web interface and get predictions for whether a transaction is fraudulent or not.

This project contains two main components:  
1. **`app.py`** – A Streamlit frontend for uploading data, training the model, and visualizing results.  
2. **`notebook.ipynb`** – Contains steps for analyzing the dataset, preprocessing features, and training the machine learning model.

---

## **Key Features**
- Upload CSV data through a web interface.
- Preprocessing of categorical data.
- Train a **Random Forest Classifier** for fraud detection.
- Predict transactions as **Fraud** or **Not Fraud**.
- Data visualizations:
  - Fraud vs Non-Fraud count  
  - Total transaction amount distribution  
  - Fraud by product type  
  - Fraud trends over time
- Download predictions as CSV.

---

## **Dataset**
- Source: Kaggle (historical online transaction data).  
- Contains information such as:
  - TransactionID  
  - Date  
  - Customer  
  - Product  
  - Quantity  
  - Price  
  - TotalAmount  
  - Fraud (target column: 1 = Fraud, 0 = Not Fraud)  

---

## **Python Libraries**
- **pandas** – For data manipulation and analysis.  
- **numpy** – For numerical computations and array operations.  
- **seaborn** – For statistical data visualization.  
- **matplotlib** – For creating plots and visualizations.  
- **scikit-learn** – For building and evaluating machine learning models.  
- **joblib** – For saving and loading trained models efficiently.  
- **streamlit** – For creating the interactive web app frontend.

Random Forest was chosen for model training due to its **high accuracy and robustness on large datasets**.

---

## **Conclusion**
- **Random Forest Classifier** performs best for detecting fraudulent and non-fraudulent transactions.  
- This tool allows **quick fraud detection** with easy-to-understand visualizations.  
- It is **flexible** and can handle any transaction dataset with similar features.
