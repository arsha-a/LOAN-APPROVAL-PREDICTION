import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model and feature list
model_path = os.path.join(os.path.dirname(__file__), "../model/best_model.pkl")
model_data = joblib.load(os.path.abspath(model_path))
model = model_data[0]
model_features = model_data[1]


st.title("Loan Approval Prediction System")
st.markdown("### Fill the form below to check your loan approval status")

# Input Fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
dependents = st.selectbox("Number of Dependents", list(range(0, 11)))
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
occupation = st.selectbox("Occupation Type", ["Salaried", "Business", "Government", "Other"])
residence = st.selectbox("Residential Status", ["Own", "Rent", "Family", "Other"])
income = st.number_input("Annual Income (₹)", min_value=0.0, step=1000.0)
expenses = st.number_input("Monthly Expenses (₹)", min_value=0.0, step=500.0)
credit_score = st.slider("Credit Score", 300, 900, 650)
existing_loans = st.selectbox("Do you have existing loans?", ["No", "Yes"])
existing_loan_amount = st.number_input("Total Existing Loan Amount (₹)", min_value=0.0, step=1000.0)
debt = st.number_input("Outstanding Debt (₹)", min_value=0.0, step=1000.0)
loan_history = st.selectbox("Loan Repayment History", ["Good", "Average", "Poor"])
loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=1000.0, step=1000.0)
loan_term = st.selectbox("Loan Term (in months)", [12, 24, 36, 60, 120, 180, 240, 300])
loan_purpose = st.selectbox("Loan Purpose", ["Education", "Personal", "Business", "Home", "Other"])
interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 10.0, step=0.5)
loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"])
coapplicant = st.selectbox("Is there a Co-Applicant?", ["No", "Yes"])
bank_history = st.selectbox("Bank Account History", ["Good", "Average", "Poor"])
transaction_freq = st.selectbox("Transaction Frequency", ["Low", "Medium", "High"])
default_risk = st.selectbox("Estimated Default Risk", ["Low", "Medium", "High"])

# Encoding
input_dict = {
    "Gender": 1 if gender == "Male" else 0,
    "Age": age,
    "Marital_Status": {"Single": 0, "Married": 1, "Divorced": 2}[marital_status],
    "Dependents": dependents,
    "Education": 1 if education == "Graduate" else 0,
    "Employment_Status": {"Employed": 0, "Unemployed": 1, "Self-Employed": 2}[employment_status],
    "Occupation_Type": {"Salaried": 0, "Business": 1, "Government": 2, "Other": 3}[occupation],
    "Residential_Status": {"Own": 0, "Rent": 1, "Family": 2, "Other": 3}[residence],
    "Annual_Income": income,
    "Monthly_Expenses": expenses,
    "Credit_Score": credit_score,
    "Existing_Loans": 1 if existing_loans == "Yes" else 0,
    "Total_Existing_Loan_Amount": existing_loan_amount,
    "Outstanding_Debt": debt,
    "Loan_History": {"Good": 0, "Average": 1, "Poor": 2}[loan_history],
    "Loan_Amount_Requested": loan_amount,
    "Loan_Term": loan_term,
    "Loan_Purpose": {"Education": 0, "Personal": 1, "Business": 2, "Home": 3, "Other": 4}[loan_purpose],
    "Interest_Rate": interest_rate,
    "Loan_Type": 1 if loan_type == "Secured" else 0,
    "Co-Applicant": 1 if coapplicant == "Yes" else 0,
    "Bank_Account_History": {"Good": 0, "Average": 1, "Poor": 2}[bank_history],
    "Transaction_Frequency": {"Low": 0, "Medium": 1, "High": 2}[transaction_freq],
    "Default_Risk": {"Low": 0, "Medium": 1, "High": 2}[default_risk],
}

input_df = pd.DataFrame([input_dict])

# Ensure all model features exist in input
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

# Prediction
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Not Approved.")

