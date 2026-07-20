import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# Load Model Files
# --------------------------------------------------
@st.cache_resource
def load_files():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, scaler, features

model, scaler, features = load_files()

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 46px;
        font-weight: 800;
        color: white;
        margin-bottom: 10px;
    }
    .sub-text {
        font-size: 18px;
        color: #cfd8dc;
        margin-bottom: 25px;
    }
    .info-box {
        background-color: #111827;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #263238;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #111827;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #263238;
        text-align: center;
    }
    .small-heading {
        font-size: 18px;
        font-weight: 600;
        color: #90caf9;
    }
    .big-text {
        font-size: 34px;
        font-weight: 700;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("💳 Credit Card Fraud Detection")

with st.expander("📊 Model Performance"):

    st.write("""
    **Algorithm:** Random Forest Classifier

    **Accuracy:** 99.95%

    **Precision:** XX%

    **Recall:** XX%

    **F1 Score:** XX%

    **ROC-AUC Score:** XX%
    """)

page = st.sidebar.radio(
    "Choose Prediction Mode",
    ["Manual Prediction", "CSV Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("ABOUT")
st.sidebar.write(
    "This application predicts whether a transaction is fraud or genuine "
    "using a random forest machine learning model."
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="main-title">💳 Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">AI-powered prediction for identifying fraudulent and genuine credit card transactions.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-box"><div class="small-heading">Model</div><div class="big-text">Random Forest</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="metric-box"><div class="small-heading">Features</div><div class="big-text">{len(features)}</div></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box"><div class="small-heading">Output</div><div class="big-text">Fraud / Genuine</div></div>', unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# Manual Prediction
# --------------------------------------------------
if page == "Manual Prediction":
    st.subheader("🔍 Manual Prediction")
    st.write("Enter transaction feature values below and click predict.")

    user_input = {}
    cols = st.columns(2)

    for i, feature in enumerate(features):
        with cols[i % 2]:
            default_value = 0.0
            if feature == "Amount":
                default_value = 100.0
            elif feature == "Time":
                default_value = 10000.0

            user_input[feature] = st.number_input(
                feature,
                value=float(default_value),
                format="%.6f"
            )

    if st.button("Predict Transaction"):
        input_df = pd.DataFrame([user_input])
        input_df = input_df[features]

        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        st.write("---")
        st.subheader("Entered Transaction")
        st.dataframe(input_df, use_container_width=True)
        st.caption(
             f"Prediction Time : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
            )

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected")
            else:
                st.success("✅ Genuine Transaction")

        with result_col2:

            st.subheader("Fraud Probability")

            st.progress(float(probability))

            st.write(f"### {probability*100:.2f}%")
     
            confidence = max(probability, 1 - probability)

            st.metric(
                "Model Confidence",
                f"{confidence*100:.2f}%"
            )

        if probability < 0.30:
            st.success("🟢 Risk Level : Low")

        elif probability < 0.70:
            st.warning("🟡 Risk Level : Medium")

        else:
            st.error("🔴 Risk Level : High")

# --------------------------------------------------
# CSV Prediction
# --------------------------------------------------
elif page == "CSV Prediction":
    st.subheader("📂 CSV Batch Prediction")
    st.write("Upload a CSV file containing transaction rows for batch prediction.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            missing_cols = [col for col in features if col not in df.columns]
            extra_cols = [col for col in df.columns if col not in features]

            if missing_cols:
                st.error("Missing required columns in the uploaded CSV:")
                st.write(missing_cols)
            else:
                if extra_cols:
                    st.warning("Extra columns found. They will be ignored.")
                    st.write(extra_cols)

                input_df = df[features].copy()
                scaled_data = scaler.transform(input_df)

                predictions = model.predict(scaled_data)
                probabilities = model.predict_proba(scaled_data)[:, 1]

                result_df = df.copy()
                result_df["Prediction"] = ["Fraud" if p == 1 else "Genuine" for p in predictions]
                result_df["Fraud_Probability"] = probabilities

                st.subheader("Prediction Results")
                st.dataframe(result_df.head(20), use_container_width=True)

                fraud_count = int((predictions == 1).sum())
                genuine_count = int((predictions == 0).sum())
                import matplotlib.pyplot as plt

                st.subheader("Prediction Summary")

                fig, ax = plt.subplots()

                ax.bar(
                    ["Fraud", "Genuine"],
                    [fraud_count, genuine_count]
                 )

                ax.set_ylabel("Transactions")

                st.pyplot(fig)

                fig2, ax2 = plt.subplots()

                ax2.pie(
                    [fraud_count, genuine_count],
                    labels=["Fraud", "Genuine"],
                    autopct="%1.1f%%"
                )

                st.pyplot(fig2)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Rows", len(result_df))
                with c2:
                    st.metric("Fraud Predictions", fraud_count)
                with c3:
                    st.metric("Genuine Predictions", genuine_count)

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error while processing file: {e}")
            st.markdown("---")

            st.caption("""
                Developed using

                Python • Streamlit • Scikit-Learn • Pandas • Joblib

                Dataset: Kaggle Credit Card Fraud Detection
                """)