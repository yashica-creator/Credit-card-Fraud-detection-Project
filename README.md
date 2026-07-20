# Credit Card Fraud Detection using Machine Learning

A machine learning-based web application that detects whether a credit card transaction is **fraudulent** or **genuine** using a trained **Random Forest Classifier**. The project also provides a simple **Streamlit web interface** for manual prediction and CSV-based batch prediction.

## Project Overview

Credit card fraud detection is a highly important classification problem because fraud transactions are extremely rare compared to normal ones. This project builds a fraud detection pipeline using machine learning on an imbalanced transaction dataset and deploys the final model through a Streamlit web app for easy usage.

The model is trained on transaction features and predicts whether a transaction belongs to:
- **0 → Genuine Transaction**
- **1 → Fraudulent Transaction**

## Dataset Information

The dataset used in this project contains **284,807 rows and 31 columns** before duplicate removal. It includes **492 fraud transactions** and **284,315 normal transactions**, which means fraud accounts for only about **0.1727%** of the data. 

During preprocessing:
- **1,081 duplicate rows** were found. 
- After removing duplicates, the dataset size became **283,726 rows and 31 columns**. 
- The updated class distribution became:
  - **283,253 genuine transactions**
  - **473 fraud transactions**
- Fraud percentage after duplicate removal became **0.1667%**.

The dataset contains the following columns:
- `Time`
- `V1` to `V28`
- `Amount`
- `Class` 

## Features

- Fraud detection using a machine learning model
- Manual transaction prediction through the Streamlit app
- CSV batch prediction for multiple transactions
- Fraud probability output
- Clean and easy-to-use interface
- Downloadable prediction results for uploaded CSV files

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Machine Learning Workflow

The project follows these major steps:

1. Data loading and exploration
2. Checking class imbalance
3. Detecting and removing duplicate rows
4. Verifying missing values and infinite values
5. Splitting data into training and testing sets
6. Feature scaling
7. Training a Random Forest Classifier
8. Evaluating model performance
9. Saving the trained model and preprocessing files
10. Deploying the prediction system using Streamlit

## Model Used

This project uses a **Random Forest Classifier** for fraud detection. Based on the notebook, the model was trained with class balancing to improve learning on the highly imbalanced dataset. 

Saved files used by the application:
- `fraud_model.pkl` → trained Random Forest model
- `scaler.pkl` → saved scaler for preprocessing
- `feature_columns.pkl` → list of feature columns expected by the model 

## Project Files

- `final(2).py` - Main Streamlit web app
- `FraudDetection(1).ipynb` - Jupyter notebook containing preprocessing, training, and evaluation
- `fraud_model.pkl` - Trained fraud detection model
- `scaler.pkl` - Saved scaler
- `feature_columns.pkl` - Saved feature column names
- `requirements.txt` - Required dependencies
- `README.md` - Project documentation
- `LICENSE` - License file 
- `Images` - Screenshots of web app

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/yashica-creator/Credit-card-Fraud-detection-Project.git
cd Credit-card-Fraud-detection-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run final(2).py
```

## How to Use the Application

### Manual Prediction
- Open the Streamlit app
- Enter the transaction feature values manually
- Click the prediction button
- The app will show whether the transaction is **Fraud** or **Genuine**

### CSV Batch Prediction
- Open the Streamlit app
- Upload a CSV file containing transaction records
- The app will process all rows and generate predictions
- Download the prediction output file

## Data Preprocessing

The notebook shows that:
- No missing values were found in the dataset columns.
- No infinite values were found in the feature matrix.
- Duplicate rows were removed before final modeling. 

This preprocessing helps make the model cleaner and more reliable.

## Problem Statement

Fraud detection is difficult because the dataset is highly imbalanced. In this dataset, fraudulent transactions form a very small fraction of the total records, so a model must learn to identify rare fraud patterns without simply predicting everything as normal. The notebook confirms this heavy imbalance in the class distribution. 

## Future Improvements

- Add **SHAP explainability** to show why a transaction was predicted as fraud
- Compare Random Forest with XGBoost, LightGBM, and Logistic Regression
- Add performance visualizations in the Streamlit app
- Deploy the app online using **Streamlit Community Cloud**
- Add threshold tuning for fraud probability
- Improve fraud detection using imbalance handling techniques such as **SMOTE**
- Add model monitoring and drift detection for real-world deployment

## Output

The application predicts:
- **Fraudulent Transaction**
- **Genuine Transaction**

It can also provide:
- Single transaction prediction
- Batch predictions from CSV input
- Prediction result download

## Learning Outcomes

This project demonstrates:
- Binary classification using machine learning
- Handling highly imbalanced fraud data
- Data preprocessing and duplicate removal
- Model training and saving
- Building an ML-powered web application using Streamlit
- End-to-end machine learning project deployment

## Author

**Yashica**  
BTech Student |GDTUW | Enrolllment No. 22301172025


**Trishla**  
BTech Student | IGDTUW | Enrolllment No. 21201172025


## License

This project is licensed under the terms provided in the repository license file.
