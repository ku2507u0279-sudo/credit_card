import json
from pathlib import Path
from datetime import time

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details below to estimate whether the transaction may be fraudulent.")
st.info("Student project demo only. Do not use this result to make real financial or security decisions.")

if not MODEL_PATH.exists():
    st.error("The trained model file `model.joblib` was not found. Train the model in Google Colab and place model.joblib in the same folder as app.py.")
    st.stop()

if not METADATA_PATH.exists():
    st.error("The file `model_metadata.json` was not found. Download it from the Colab notebook and place it in the same folder as app.py.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

locations = metadata.get("locations", ["New York"])
transaction_types = metadata.get("transaction_types", ["purchase", "refund"])

st.subheader("Transaction details")
with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        TransactionID = st.number_input(
            "TransactionID", min_value=0, value=100001, step=1,
            help="Transaction identifier. Use a value in the same format as the dataset."
        )
        Amount = st.number_input(
            "Amount", min_value=0.0, value=250.00, step=10.0, format="%.2f"
        )
        MerchantID = st.number_input(
            "MerchantID", min_value=0, value=1, step=1
        )

    with col2:
        transaction_date = st.date_input(
            "TransactionDate",
            value=pd.Timestamp("2024-10-21").date(),
            help="Choose the transaction date."
        )
        transaction_time = st.time_input(
            "Transaction time",
            value=time(12, 0),
            help="Choose the transaction time."
        )
        TransactionType = st.selectbox("TransactionType", transaction_types)
        Location = st.selectbox("Location", locations)

    submitted = st.form_submit_button("🔍 Check transaction", use_container_width=True)

if submitted:
    # The visible input labels match the dataset column names exactly.
    transaction_datetime = pd.Timestamp.combine(transaction_date, transaction_time)
    raw_input = pd.DataFrame([{
        "TransactionID": int(TransactionID),
        "TransactionDate": transaction_datetime,
        "Amount": float(Amount),
        "MerchantID": int(MerchantID),
        "TransactionType": str(TransactionType),
        "Location": str(Location)
    }])

    # Match the feature engineering used in the Colab notebook.
    features = raw_input.copy()
    features["TransactionDate"] = pd.to_datetime(features["TransactionDate"], errors="coerce")
    features["TransactionYear"] = features["TransactionDate"].dt.year
    features["TransactionMonth"] = features["TransactionDate"].dt.month
    features["TransactionDay"] = features["TransactionDate"].dt.day
    features["TransactionHour"] = features["TransactionDate"].dt.hour
    features["TransactionDayOfWeek"] = features["TransactionDate"].dt.dayofweek
    features = features.drop(columns=["TransactionDate"])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    fraud_index = list(model.classes_).index(1)
    fraud_probability = float(probabilities[fraud_index])

    st.divider()
    st.subheader("Prediction result")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Estimated fraud probability", f"{fraud_probability:.1%}")
    metric_col2.metric("Model classification", "Fraud" if prediction == 1 else "Not Fraud")

    if prediction == 1:
        st.error("⚠️ The model classifies this transaction as potentially fraudulent.")
    else:
        st.success("✅ The model classifies this transaction as not fraudulent.")

    st.caption("This is a model estimate, not a guarantee. The displayed probability may not be well calibrated.")
    with st.expander("View the input sent to the model"):
        st.dataframe(raw_input, use_container_width=True)
