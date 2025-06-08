import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("fraud_model.json")  

def predict(data):
    return model.predict(data)

st.title("💳 Fraud Detection App")

st.markdown("""
Upload a **CSV file** with 31 features (like `creditcard.csv`) to check for fraud.

Required columns: `Time`, `V1` to `V28`, `Amount`, `Hour`

 Download sample dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
""")

uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # If 'Hour' column not there, create it from 'Time'
    if 'Hour' not in data.columns:
        data['Hour'] = (data['Time'] // 3600) % 24

    # If 'Class' is there (actual labels), remove it
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    # When user clicks "Predict" button
    if st.button("Predict"):
        # Check if file has exactly 31 columns (as expected by model)
        if data.shape[1] == 31:
            predictions = predict(data)  # Make prediction
            data['Prediction'] = predictions  # Add predictions to the data

            # Filter only predicted frauds
            frauds = data[data['Prediction'] == 1]

            # Show results
            if not frauds.empty:
                st.write(f" Found **{len(frauds)}** fraud transactions out of {len(data)}.")
                percent = (len(frauds) / len(data)) * 100
                st.write(f" Fraud percentage: {percent:.2f}%")
                st.dataframe(frauds.drop(columns=['Prediction']))  # Show frauds only
            else:
                st.write(" No fraud found in your data! Party safe!")
        else:
            st.error(f" File has {data.shape[1]} columns. It should have 31 for prediction to work.")
