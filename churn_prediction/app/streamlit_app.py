"""
Customer Churn Prediction App
==============================
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = pickle.load(open("models/xgb_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    features = pickle.load(open("models/feature_names.pkl", "rb"))
    return model, scaler, features

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🔮 Customer Churn Prediction")
st.markdown("**Predict whether a telecom customer is likely to churn based on their profile.**")
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR - INPUT FORM
# ─────────────────────────────────────────
st.sidebar.header("📋 Customer Details")

# Basic Info
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents     = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)

# Services
st.sidebar.subheader("📡 Services")
phone_service    = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines   = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_security  = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup    = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection= st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support     = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv     = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Account Info
st.sidebar.subheader("💳 Account Info")
contract          = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method    = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges   = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
total_charges     = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, monthly_charges * tenure, step=10.0)

# ─────────────────────────────────────────
# PREPARE INPUT
# ─────────────────────────────────────────
def prepare_input(features):
    """Build a dataframe matching the trained model's feature set."""
    row = {f: 0 for f in features}

    # Direct numerics
    row['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    row['tenure']        = tenure
    row['MonthlyCharges']= monthly_charges
    row['TotalCharges']  = total_charges

    # Binary
    row['gender']           = 1 if gender == "Male" else 0
    row['Partner']          = 1 if partner == "Yes" else 0
    row['Dependents']       = 1 if dependents == "Yes" else 0
    row['PhoneService']     = 1 if phone_service == "Yes" else 0
    row['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0

    # Feature engineering
    row['charges_per_month']   = total_charges / (tenure + 1)
    row['high_value_customer'] = 1 if monthly_charges > 64.76 else 0

    # One-hot encoded columns (match exact column names from get_dummies)
    def set_ohe(prefix, value, options):
        for opt in options[1:]:  # drop_first=True skips first
            col = f"{prefix}_{opt}"
            if col in row:
                row[col] = 1 if value == opt else 0

    set_ohe("MultipleLines",   multiple_lines,    ["No", "No phone service", "Yes"])
    set_ohe("InternetService", internet_service,  ["DSL", "Fiber optic", "No"])
    set_ohe("OnlineSecurity",  online_security,   ["No", "No internet service", "Yes"])
    set_ohe("OnlineBackup",    online_backup,     ["No", "No internet service", "Yes"])
    set_ohe("DeviceProtection",device_protection, ["No", "No internet service", "Yes"])
    set_ohe("TechSupport",     tech_support,      ["No", "No internet service", "Yes"])
    set_ohe("StreamingTV",     streaming_tv,      ["No", "No internet service", "Yes"])
    set_ohe("StreamingMovies", streaming_movies,  ["No", "No internet service", "Yes"])
    set_ohe("Contract",        contract,          ["Month-to-month", "One year", "Two year"])
    set_ohe("PaymentMethod",   payment_method,    [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"
    ])

    return pd.DataFrame([row])[features]


# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

st.markdown("---")

if predict_btn:
    try:
        model, scaler, features = load_model()
        input_df = prepare_input(features)
        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        # Result display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Prediction Result")
            if pred == 1:
                st.error(f"⚠️ **HIGH CHURN RISK**")
                st.metric("Churn Probability", f"{prob*100:.1f}%", delta="At Risk")
            else:
                st.success(f"✅ **LOW CHURN RISK**")
                st.metric("Churn Probability", f"{prob*100:.1f}%", delta="Likely to Stay")

        with col2:
            st.subheader("📊 Risk Gauge")
            # Color based on risk
            if prob < 0.3:
                color = "green"
                label = "Low Risk"
            elif prob < 0.6:
                color = "orange"
                label = "Medium Risk"
            else:
                color = "red"
                label = "High Risk"

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border-left: 5px solid {color};
            ">
                <h1 style="color:{color}; font-size:48px; margin:0">{prob*100:.0f}%</h1>
                <p style="font-size:18px; color:#555; margin:5px 0">{label}</p>
                <p style="font-size:13px; color:#888">Churn Probability</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Recommendations
        st.subheader("💡 Business Recommendations")
        if prob > 0.6:
            st.markdown("""
            - 🎁 **Offer a loyalty discount** or promotional deal immediately
            - 📞 **Proactive outreach** — assign customer success manager
            - 📦 **Upgrade contract** to yearly with incentive
            - 🛡️ **Bundle additional services** to increase stickiness
            """)
        elif prob > 0.3:
            st.markdown("""
            - 📧 **Send engagement emails** with product tips
            - 🎯 **Offer tech support or onboarding** assistance
            - 📊 **Monitor usage patterns** over next 30 days
            """)
        else:
            st.markdown("""
            - ✅ Customer appears satisfied — maintain service quality
            - 🌟 **Upsell opportunity** — consider premium plan offer
            - 📣 **Referral program** — happy customers refer others
            """)

    except FileNotFoundError:
        st.error("❌ Model not found! Please run `train_model.py` first to train the model.")

else:
    # Show instructions
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn**")

    # Show sample plots if they exist
    if os.path.exists("models/feature_importance.png"):
        st.subheader("📊 Model Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.image("models/feature_importance.png", caption="Top Features")
        with col2:
            st.image("models/roc_curve.png", caption="ROC Curve")
