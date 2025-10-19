%%writefile app.py
# app.py (6-feature version)
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Breast Cancer (6-feature) Prediction", layout="centered")

st.title("🩺 Breast Cancer Prediction — Top 6 Features")
st.write("This app uses 6 important features to predict benign vs malignant (educational demo).")

MODEL_PATH = "rf_model_top6.joblib"
FEATURES_PATH = "feature_names_top6.csv"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Run retrain_top6.py to create it.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_names = pd.read_csv(FEATURES_PATH, header=None).squeeze().tolist()

st.sidebar.header("Enter feature values (6 features)")
inputs = []
default_vals = [15.0, 0.05, 0.05, 95.0, 600.0, 0.05]

inputs = []
for feature in feature_names:
    val = st.number_input(f"Enter value for {feature}:", min_value=0.0, max_value=1000.0, step=0.01)
    inputs.append(val)


st.subheader("Input values")
input_df = pd.DataFrame([inputs], columns=feature_names)
st.dataframe(input_df.T.rename(columns={0: "value"}))

if st.button("Predict"):
    arr = np.array(inputs).reshape(1, -1)
    pred = model.predict(arr)[0]
    proba = model.predict_proba(arr)[0]
    label = "Benign (No Cancer)" if pred == 1 else "Malignant (Cancer)"
    st.write(f"**Prediction:** {label}")
    st.write(f"Benign probability: {proba[1]*100:.2f}% | Malignant probability: {proba[0]*100:.2f}%")
    st.info("⚠️ Educational demo — not for medical use.")
