import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("bankruptcy_model.pkl")

st.title("💼 Bankruptcy Prediction App")
st.write("Enter the business risk factors to predict the probability of bankruptcy.")

# Input fields
industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
credibility = st.selectbox("Credibility", [0, 0.5, 1])
competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])

# Predict button
if st.button("Predict Bankruptcy Status"):
    input_data = np.array([[industrial_risk, management_risk, financial_flexibility,
                            credibility, competitiveness, operating_risk]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of bankruptcy
    
    if prediction == 1:
        st.error(f"⚠️ The business is likely to go BANKRUPT (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The business is NOT likely to go bankrupt (Probability: {probability:.2f})")
