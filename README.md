# Credit Card Fraud Detection System

A machine learning web application that predicts whether a credit card transaction is **fraudulent** or **genuine** using a trained Random Forest model.

## Project Overview

This project is built to detect fraudulent credit card transactions using machine learning and deploy the prediction system through a Streamlit web application. The app supports both **manual prediction** by entering transaction details and **CSV batch prediction** by uploading transaction data files. A good README for an ML project should explain the project goal, usage, data, file structure, and setup clearly, which this format follows. [web:333][web:337]

## Features

- Manual transaction prediction
- CSV batch prediction
- Fraud probability score
- Clean Streamlit user interface
- Downloadable prediction results for uploaded CSV files

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Model

This project uses a **Random Forest Classifier** for fraud detection. The model is trained on processed transaction features, and the app loads the saved model files to make predictions on new input data.

## Project Files

- `final.py` - Main Streamlit application file
- `fraud_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Saved scaler used for preprocessing
- `feature_columns.pkl` - Feature column names used by the model
- `requirements.txt` - Required Python libraries
- `sample_transactions.csv` - Sample CSV file for testing
- `README.md` - Project documentation
- `LICENSE` - MIT License

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/yashica-creator/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run final.py
```

## How to Use the App

### Manual Prediction
Choose **Manual Prediction** from the sidebar and enter the transaction feature values. Then click the prediction button to check whether the transaction is fraud or genuine.

### CSV Prediction
Choose **CSV Prediction** from the sidebar and upload a CSV file containing transaction rows. The app will generate predictions for all rows and allow you to download the results.

## Dataset

This project is based on a credit card fraud detection dataset used for machine learning classification. If you used a public dataset such as Kaggle, include its link here because dataset source information is an important README section for reproducibility. [web:333][web:338]

Example format:

- Dataset source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Output

The application predicts:
- **Fraud**
- **Genuine**

It also shows:
- Fraud probability score
- Batch prediction results for CSV uploads
- Download option for prediction output

## Future Improvements

- Add SHAP explainability for predictions
- Improve visual dashboard
- Deploy the app online using Streamlit Community Cloud
- Add model performance metrics such as accuracy, precision, recall, and F1-score

## Author

1.YASHICA 
BTech Student | Machine Learning Project

2.TRISHLA
BTech Student | Machine Learning Project

## License

This project is licensed under the MIT License.
