"""
Telco Customer Churn — Prediction Demo (Streamlit)
====================================================

Loads the model artifacts saved by the notebook (churn_best_model.pkl,
churn_scaler.pkl, churn_model_columns.pkl) and lets you enter one
customer's details to see the predicted churn probability live.

This is the piece the project documentation refers to under
"Machine Learning and Model Evaluation (Also showcase streamlit app)".

HOW TO RUN
----------
1. Put this file in the SAME folder as:
     churn_best_model.pkl
     churn_scaler.pkl
     churn_model_columns.pkl
   (these are created by running the modeling notebook to the end)

2. Install Streamlit (one-time):
     pip install streamlit

3. Run it:
     streamlit run app.py

   This opens a local web page in your browser — that's the "web app"
   demo for the ML part of the presentation.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="centered"
)

# ----------------------------------------------------------------------
# Load the saved model artifacts (cached so they only load once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_best_model.pkl")
    scaler = joblib.load("churn_scaler.pkl")
    model_columns = joblib.load("churn_model_columns.pkl")
    try:
        model_name = joblib.load("churn_best_model_name.pkl")
    except FileNotFoundError:
        model_name = type(model).__name__
    return model, scaler, model_columns, model_name


try:
    model, scaler, model_columns, model_name = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Couldn't find the model files. Make sure 'churn_best_model.pkl', "
        "'churn_scaler.pkl' and 'churn_model_columns.pkl' are in the same "
        "folder as this app, then restart it.\n\n"
        f"Details: {e}"
    )
    st.stop()

# These must exactly match what the training notebook used.
CATEGORICAL_COLS = [
    'city', 'gender', 'senior_citizen', 'partner', 'dependents',
    'phone_service', 'multiple_lines', 'internet_service', 'online_security',
    'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
    'streaming_movies', 'contract', 'paperless_billing', 'payment_method'
]
NUMERIC_COLS = ['tenure', 'monthly_charges', 'total_charges']

CITY_OPTIONS = [
    'Alexandria', 'Banha', 'Cairo', 'Damietta', 'El Mahalla El Kubra',
    'Faiyum', 'Giza', 'Mansoura', 'Port Said', 'Sharm El Sheikh',
    'Suez', 'Tanta', 'Zagazig'
]

# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
st.title("📉 Telco Customer Churn Predictor")
st.caption(f"Model in use: **{model_name}**")
st.write("Enter a customer's details below to estimate their probability of churning.")

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("customer_form"):
    st.subheader("Customer Profile")
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", CITY_OPTIONS)
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    col3, col4 = st.columns(2)
    with col3:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
    with col4:
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)

    submitted = st.form_submit_button("Predict Churn Probability")

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    customer = {
        'city': city,
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
    }

    row = pd.DataFrame([customer])

    # Same encoding approach used during training: one-hot encode, then
    # align to the exact columns the model was trained on (missing dummy
    # columns for categories not chosen here are filled with 0).
    row_encoded = pd.get_dummies(row, columns=CATEGORICAL_COLS)
    row_encoded = row_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale the numeric columns with the SAME scaler fitted during training.
    row_encoded[NUMERIC_COLS] = scaler.transform(row[NUMERIC_COLS])

    probability = float(model.predict_proba(row_encoded)[:, 1][0])

    st.divider()
    st.subheader("Result")

    st.metric("Predicted Churn Probability", f"{probability * 100:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))

    if probability >= 0.6:
        st.error("🔴 High risk — this customer is likely to churn. Consider proactive retention action.")
    elif probability >= 0.3:
        st.warning("🟠 Medium risk — worth monitoring or a light-touch retention offer.")
    else:
        st.success("🟢 Low risk — this customer looks likely to stay.")

    with st.expander("See the exact inputs used"):
        st.json(customer)

st.divider()
st.caption(
    "This app loads the model trained in the modeling notebook and applies the same "
    "encoding/scaling steps to a single new customer, so the probability shown here is "
    "produced the same way as the test-set evaluation in the notebook."
)
