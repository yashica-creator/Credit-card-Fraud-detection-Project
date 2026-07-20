# Credit Card Fraud Detection System

A machine learning-based web application for detecting fraudulent credit card transactions using a trained **Random Forest Classifier**. The project includes both **single transaction prediction** and **CSV batch prediction** through an interactive Streamlit interface, along with fraud probability, confidence score, risk level indication, and downloadable output results.

---

## Overview

Credit card fraud is a major issue in digital payments, and early identification of suspicious transactions helps reduce financial loss. This project uses a machine learning model to classify a transaction as either **Fraudulent** or **Genuine** based on transaction features, and it exposes the model through a user-friendly Streamlit web app for practical use. 

The application supports:
- Manual entry of feature values for single prediction
- CSV upload for batch prediction
- Fraud probability display
- Model confidence display
- Risk level classification
- Summary charts for batch predictions
- Downloadable CSV output with predictions

---

## Features

- **Manual Prediction**
  Enter transaction feature values one by one and predict whether the transaction is fraud or genuine. 

- **CSV Batch Prediction**
  Upload a CSV file containing multiple transaction rows and generate predictions for all transactions at once. 

- **Fraud Probability**
  Displays the probability score returned by the trained model for the fraud class.

- **Model Confidence**
  Shows confidence based on the higher class probability for the final prediction.

- **Risk Level Indicator**
  The app classifies the result into:
  - Low risk
  - Medium risk
  - High risk

- **Prediction Summary Charts**
  Batch mode includes a bar chart and pie chart showing counts of fraud and genuine predictions. 

- **Download Results**
  Users can download batch prediction results as a CSV file.

---

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **Scikit-learn**
- **Joblib**
- **Matplotlib** 

---

## Machine Learning Workflow

The machine learning pipeline used in this project follows a standard fraud detection workflow:

1. Load and inspect the dataset.
2. Separate features and target variable.
3. Split the dataset into train and test sets using stratified sampling.
4. Scale the input features using `StandardScaler`.
5. Handle class imbalance using **SMOTE** on the training data.
6. Train classification models.
7. Evaluate model performance using confusion matrix, classification report, ROC-AUC, and PR-AUC.
8. Save the final trained model and preprocessing files for deployment. 

The notebook shows:
- Dataset shape of `283726 x 30`
- Fraud percentage of about `0.1727%`
- Stratified train-test split
- SMOTE-based balancing of the training set
- Evaluation using classification metrics including ROC-AUC and PR-AUC 

---

## Model Used

This project uses a **Random Forest Classifier** in the deployed Streamlit application. The app loads:
- `fraud_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl` 

The sidebar in the app also identifies the model as **Random Forest Classifier**.

---

## Project Structure

```bash
Credit-Card-Fraud-Detection/
│
├── webapp.py
├── FraudDetection(1).ipynb
├── fraud_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── sample_transactions.csv
├── README.md
└── LICENSE
```

### File Description

- `webapp.py` — Main Streamlit web application for prediction and UI. 
- `FraudDetection(1).ipynb` — Jupyter notebook containing data preprocessing, SMOTE balancing, training, and evaluation workflow. 
- `fraud_model.pkl` — Saved trained Random Forest model used by the app. 
- `scaler.pkl` — Saved scaler used to transform input data before prediction. 
- `feature_columns.pkl` — Saved list of feature columns expected by the model. 
- `requirements.txt` — Python dependencies required to run the project.
- `sample_transactions.csv` — Example file for testing batch prediction.
- `README.md` — Project documentation.
- `LICENSE` — Repository license file.

---

## Dataset

This project is based on a **credit card fraud detection dataset** containing highly imbalanced transaction classes, where fraud cases form only a very small percentage of the total data. In the notebook, the fraud percentage is shown as approximately `0.1727%`, which confirms the imbalance problem and justifies the use of SMOTE during training. 

The notebook also shows:
- Total processed samples: `283726`
- Total features used for training: `30`
- Train size: `226980`
- Test size: `56746`

### Suggested dataset link

- [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


---

## Data Imbalance Handling

Fraud detection datasets are typically extremely imbalanced, which can cause models to favor the majority class. In this project, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the scaled training data to balance the fraud and genuine classes before model training. 

After applying SMOTE in the notebook:
- Class `0`: `226602`
- Class `1`: `226602` 

This helps the model learn fraud patterns more effectively than training on the original imbalanced distribution alone.

---

## Model Evaluation

The notebook includes evaluation using:
- Confusion Matrix
- Classification Report
- ROC-AUC
- PR-AUC 

One of the evaluated results shown in the notebook includes:
- ROC-AUC: `0.9626`
- PR-AUC: `0.6750` 

Because fraud detection is an imbalanced classification problem, **PR-AUC, recall, and confusion-matrix-level analysis** are especially important in addition to accuracy.

---

## Web Application Interface

The Streamlit application provides two prediction modes through the sidebar:

### 1. Manual Prediction
In this mode, users enter values for each feature manually and click the **Predict Transaction** button. The app then:
- preprocesses the input,
- scales it using the saved scaler,
- predicts the class,
- shows fraud probability,
- shows model confidence,
- displays risk level,
- and prints the entered transaction data. 

### 2. CSV Prediction
In this mode, users upload a CSV file with the required columns. The app then:
- validates the uploaded file,
- detects missing and extra columns,
- runs batch predictions,
- displays prediction results,
- shows summary metrics,
- generates bar and pie charts,
- and provides a CSV download button for the results. 

---

## Risk Level Logic

The deployed app categorizes fraud probability into risk levels as follows:

- **Low Risk**: probability below `0.30`
- **Medium Risk**: probability below `0.70`
- **High Risk**: probability `0.70` or above

This makes the output easier to interpret for non-technical users.

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/yashica-creator/Credit-card-Fraud-detection-Project.git
cd Credit-card-Fraud-detection-Project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

#### Activate it on Windows
```bash
venv\Scripts\activate
```

#### Activate it on macOS/Linux
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run webapp.py
```

After running the command, Streamlit will open the app in your browser.

---

## How to Use

### Manual Prediction

1. Launch the app.
2. Select **Manual Prediction** from the sidebar.
3. Enter values for all required transaction features.
4. Click **Predict Transaction**.
5. View:
   - prediction result,
   - fraud probability,
   - model confidence,
   - risk level,
   - and entered transaction details. 

### CSV Batch Prediction

1. Launch the app.
2. Select **CSV Prediction** from the sidebar.
3. Upload a CSV file containing the required feature columns.
4. Review the uploaded data preview.
5. View:
   - fraud/genuine prediction for each row,
   - fraud probability values,
   - summary metrics,
   - bar chart,
   - pie chart.
6. Download the results CSV using the provided button.

---

## Expected Input Format

The uploaded CSV file should contain the same feature columns used during model training. The app checks the uploaded file against the stored `feature_columns.pkl` and reports:
- missing columns as errors,
- extra columns as warnings,
- and ignores extra columns during prediction. 

This ensures consistency between training-time and inference-time input format. 

---

## Output Format

For each transaction, the system predicts:
- **Fraud**
- **Genuine** 

In batch mode, the downloadable CSV includes:
- original input columns,
- predicted label,
- fraud probability score.

---

## Screenshots


### Home Page
```md

```

### Manual Prediction
```md

```

### CSV Prediction
```md

```

### Results Dashboard
```md

```

---

## Future Improvements

This project can be extended further by adding:

- SHAP-based explainability for individual predictions
- Better fraud analytics dashboard
- Additional models such as XGBoost or LightGBM
- Model comparison section in the web app
- Threshold tuning for fraud sensitivity
- Real-time transaction API deployment
- User authentication and analyst dashboard
- Drift detection and model monitoring for production systems

---

## Learning Outcomes

Through this project, the following concepts were applied:

- Imbalanced classification
- SMOTE oversampling
- Data preprocessing and feature scaling
- Fraud detection model training
- Classification metric evaluation
- Model serialization with Joblib
- Streamlit-based ML deployment
- Batch inference using CSV input 

---

## Installation Requirements

A typical `requirements.txt` for this project may include:

```txt
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
imbalanced-learn
```

Make sure your local `requirements.txt` matches the libraries actually used in your environment.

---

## Deployment

This project can be deployed on:

- **Streamlit Community Cloud**
- **Render**
- **Railway**
- **Hugging Face Spaces** for lightweight demos

---

## Author

**Yashica**  
BTech Student | IGDTUW | Enrollment No. 22301172025


**Trishla**  
BTech Student | IGDTUW | Enrollment No. 21201172025


---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgements

- Kaggle / public fraud detection dataset source
- Scikit-learn for machine learning utilities
- Streamlit for web deployment
- Imbalanced-learn for SMOTE implementation [file:386][file:407]
