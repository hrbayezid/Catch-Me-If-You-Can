from xgboost import XGBClassifier
import pandas as pd
import streamlit as st

# Load trained model
model = XGBClassifier()
model.load_model("fraud_model.json")  # 🧠 This works only if it was fitted and saved like above!

# Prediction function
def predict(data):
    return model.predict(data)

# Streamlit UI
st.title("💳 Fraud Detection App")
uploaded_file = st.file_uploader("Upload transaction CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Make sure 'Hour' exists (generate if not)
    if 'Hour' not in data.columns:
        data['Hour'] = (data['Time'] // 3600) % 24

    # Drop 'Class' if user uploaded with it
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    if st.button("Predict"):
        if data.shape[1] == 31:
            predictions = predict(data)
            
            # Add predictions as a new column to the dataframe
            data['Prediction'] = predictions
            
            # Filter only frauds (Prediction == 1)
            frauds = data[data['Prediction'] == 1]
            
            # Show frauds if any found
            if not frauds.empty:
                st.write(f"🚨 Detected **{len(frauds)}** fraud transactions out of {len(data)} total.")
                fraud_percentage = (len(frauds) / len(data)) * 100
                st.write(f"⚠️ Fraud percentage: {fraud_percentage:.2f}%")
                st.dataframe(frauds.drop(columns=['Prediction']))  # show features only, no prediction col
                
            else:
                st.write("🎉 No fraud detected in the uploaded transactions!")
        else:
            st.error(f"Uploaded file has {data.shape[1]} columns, expected 31.")
