import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('fraud_model.joblib')

st.title("🚨 Fraud Detection App")


# Upload CSV
uploaded_file = st.file_uploader("Upload your transaction CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(data.head())

    if st.button("Detect Fraud!"):
        # Remove target if it’s there
        X_new = data.drop('Class', axis=1, errors='ignore')
        predictions = model.predict(X_new)
        data['Fraud_Prediction'] = predictions
        st.write("Prediction results:")
        st.dataframe(data)
