import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("rf_model.joblib")
feature_names = pd.read_csv("feature_names.csv", header=None)[0].tolist()

st.title("🔬 Breast Cancer Prediction App")

# User input
input_data = []
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}:", min_value=0.0, step=0.1)
    input_data.append(value)

# Prediction button
if st.button("Predict"):
    prediction = model.predict([input_data])
    result = "Benign (No Cancer)" if prediction[0] == 1 else "Malignant (Cancer)"
    st.success(f"Prediction: {result}")
