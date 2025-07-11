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
    
    # **THE FIX IS HERE:**
    # The column names in this list MUST be IDENTICAL to the column names
    # used when training the model. Check for case-sensitivity, spaces, or underscores.
    # For example, if your training data used "annual income" (lowercase), you must use that here.
    
    expected_columns = ["Age", "Annual_Income", "Employment_Status", "Num_Bank_Accounts", "Total_EMI_per_month"]
    
    # Create a dictionary to hold the user's input
    input_data = {
        "Age": [age],
        "Annual_Income": [income],
        "Employment_Status": [employment_map[employment]], # Convert text to number
        "Num_Bank_Accounts": [bank_accounts],
        "Total_EMI_per_month": [emi]
    }
    
    # Create the DataFrame using the dictionary and ensure column order
    input_df = pd.DataFrame(input_data)
    input_df = input_df[expected_columns] # This enforces the correct column order

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

        # Optional: Uncomment to see the data being sent to the model for debugging
        # with st.expander("Show Debug Info"):
        #     st.write("Input DataFrame:", input_df)
        #     st.write("Scaled Input:", input_scaled)
        #     st.write("Predicted Score (0=High Risk, 1=Medium, 2=Low):", prediction)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure the input values are correct. If the error persists, the model's expected features might not match the app's inputs.")

