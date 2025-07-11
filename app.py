import streamlit as st
import pandas as pd
from joblib import load

# --- 1. Load Artifacts ---
# Load the trained model and the scaler object.
# Ensure these files are in the same directory as your app.py, or provide the full path.
try:
    model = load("credit_model_small_compressed.joblib")
    scaler = load("scaler_new.joblib")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'credit_model_small_compressed.joblib' and 'scaler_new.joblib' are in the correct folder.")
    st.stop()


# --- 2. Define Mappings and Helper Functions ---
# This map converts the user-friendly dropdown selection into the numerical format the model expects.
employment_map = {
    "Employed": 1,
    "Unemployed": 2,
    "Self-Employed": 0
}

# This function determines the final output message based on the model's prediction.
def loan_decision(prediction_score):
    """Returns a user-friendly loan decision string based on the model's output."""
    if prediction_score == 2:
        return "✅ APPROVED — Low Risk"
    elif prediction_score == 1:
        return "⚠️ REVIEW — Medium Risk"
    else:
        return "❌ REJECT — High Risk"

# --- 3. Set Up The User Interface (UI) ---
st.set_page_config(page_title="AI Credit Scoring", layout="centered")
st.title("💳 AI Credit Scoring & Loan Approval")
st.markdown(
    "This app uses a machine learning model to predict loan approval risk. "
    "Fill in the applicant's details below to get a prediction."
)
st.divider()


# --- 4. Get User Input ---
# Create input fields for all the features your model needs.
# The `value` parameter sets a default, which is helpful for quick testing.
age = st.slider("Age", min_value=18, max_value=100, value=30, help="Applicant's age.")
income = st.number_input("Annual Income (₹)", min_value=10000.0, value=500000.0, step=10000.0, help="Applicant's total annual income.")
employment = st.selectbox("Employment Status", options=["Employed", "Unemployed", "Self-Employed"], help="Applicant's current employment status.")
bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=2, help="Total number of bank accounts the applicant holds.")
emi = st.number_input("Total Monthly EMI (₹)", min_value=0.0, value=10000.0, step=500.0, help="Total existing monthly EMI payments.")


# --- 5. Prediction Logic ---
# This block runs only when the user clicks the "Predict" button.
if st.button("Predict Loan Approval", type="primary"):
    
    # **FINAL FIX: PASTE YOUR CORRECT COLUMN NAMES HERE**
    # Run the `check_features.py` script to get the exact list of column names.
    # Then, paste that list here. It must be 100% identical to the output of the script.
    # EXAMPLE: expected_columns = ['age', 'annual_income', 'employment_status', 'num_bank_accounts', 'total_emi']
    
    expected_columns = ["Age", "Annual_Income", "Employment_Status", "Num_Bank_Accounts", "Total_EMI_per_month"] # <-- REPLACE THIS LIST
    
    # Create a dictionary to hold the user's input.
    # The keys here don't matter as much, but it's good practice to keep them clear.
    input_data = {
        expected_columns[0]: [age],
        expected_columns[1]: [income],
        expected_columns[2]: [employment_map[employment]],
        expected_columns[3]: [bank_accounts],
        expected_columns[4]: [emi]
    }
    
    # Create the DataFrame
    input_df = pd.DataFrame(input_data)
    
    # This is the crucial step: ensure the DataFrame has the exact columns in the exact order.
    # This line is no longer strictly necessary if the dictionary keys are correct, but it's a good safeguard.
    input_df = input_df[expected_columns]

    try:
        # Scale the input features using the pre-fitted scaler
        input_scaled = scaler.transform(input_df)

        # Make a prediction using the trained model
        prediction = model.predict(input_scaled)[0]
        
        # Get the user-friendly decision text
        decision = loan_decision(prediction)

        # --- 6. Display Output ---
        st.subheader("Loan Decision Result")
        
        # Display the result in a styled box
        if "APPROVED" in decision:
            st.success(decision)
        elif "REVIEW" in decision:
            st.warning(decision)
        else:
            st.error(decision)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please double-check that the `expected_columns` list in the code is 100% correct.")

