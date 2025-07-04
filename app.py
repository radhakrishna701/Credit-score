import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load model and scaler
model = load("credit_model_compressed.joblib")
scaler = load("scaler.joblib")

# Encode Employment manually (simple logic)
employment_map = {
    "Employed": 1,
    "Unemployed": 0,
    "Self-Employed": 2
}

# Loan decision logic
def loan_decision(score):
    if score == 2:
        return "✅ APPROVED — Low Risk"
    elif score == 1:
        return "⚠️ REVIEW — Medium Risk"
    else:
        return "❌ REJECT — High Risk"

# Streamlit UI
st.title("AI Credit Risk & Loan Approval App")

st.write("Enter your financial and personal info below to get a loan approval decision:")

age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Annual Income (in ₹)", min_value=0.0, value=300000.0)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=2)
loan_required = st.number_input("Loan Amount Required (Estimate)", min_value=1000.0, value=50000.0)

if st.button("Check Loan Risk"):
    # Create input vector (dummy format based on trained model expectations)
    input_data = pd.DataFrame([[
        age, income, employment_map[employment_status], bank_accounts, loan_required
    ]], columns=["Age", "Annual_Income", "Employment_Status", "Num_Bank_Accounts", "Loan_Amount"])

    # Manually scale and format to match model
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Loan decision output
    decision = loan_decision(prediction)

    st.subheader(f"Prediction: {decision}")
