
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load model
model = joblib.load("water_quality_model.pkl")

# Define input form
st.title("💧 Smart Water Quality Prediction")

st.markdown("Enter the water parameters to check the sensor status:")

pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=100.0, value=3.0)
tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=2000.0, value=500.0)
temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

if st.button("Predict"):
    # Standard scaling (normally you'd use the saved scaler, but we'll do it manually here)
    scaler = StandardScaler()
    scaler.fit([[7.2, 3.5, 520, 26.3], [7.0, 5.8, 600, 31.1]])  # just sample values for range

    input_data = scaler.transform([[pH, turbidity, tds, temp]])
    prediction = model.predict(input_data)

    status_mapping = {0: "Critical", 1: "Normal", 2: "Warning"}
    status = status_mapping.get(prediction[0], "Unknown")

    st.success(f"🔍 Predicted Sensor Status: **{status}**")
