import streamlit as st
import pandas as pd
from joblib import load

# Load model and scaler
model = load("credit_model_small_compressed.joblib")
scaler = load("scaler_new.joblib")

# Employment status map
employment_map = {
    "Employed": 1,
    "Unemployed": 2,
    "Self-Employed": 0
}

# Decision logic
def loan_decision(score):
    if score == 2:
        return "✅ APPROVED — Low Risk"
    elif score == 1:
        return "⚠️ REVIEW — Medium Risk"
    else:
        return "❌ REJECT — High Risk"

# UI
st.title("💳 AI Credit Scoring & Loan Approval App")
st.markdown("Fill in the information below to check loan approval based on your credit score.")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (₹)", min_value=10000.0, value=500000.0)
employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=2)
emi = st.number_input("Monthly EMI Required (₹)", min_value=0.0, value=10000.0)

# Predict
if st.button("Predict Loan Approval"):
    # Match feature order used in training
    input_df = pd.DataFrame([[
        age,
        income,
        employment_map[employment],
        bank_accounts,
        emi
    ]], columns=["Age", "Annual_Income", "Employment_Status", "Num_Bank_Accounts", "Total_EMI_per_month"])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict score
    score = model.predict(input_scaled)[0]
    decision = loan_decision(score)

    # Output
    st.subheader("Loan Decision Result")
    st.success(decision)

    # Optional debug
    # st.write("Input DataFrame:", input_df)
    # st.write("Scaled Input:", input_scaled)
    # st.write("Predicted Score:", score)
