import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set background and logo
def set_bg_and_logo():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('/Users/arun/Downloads/cards111.png');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="logo-container"><img src="/Users/arun/Downloads/logo.png" width="150"></div>', unsafe_allow_html=True)

# Call background and logo
set_bg_and_logo()

# Title
st.title("💳 Credit Card Fraud Detection")

# Input fields
amount = st.number_input("Enter Transaction Amount", min_value=0.0, format="%.2f")

# AM/PM time input
col1, col2 = st.columns(2)

with col1:
    hour = st.selectbox("Select Hour (1–12)", list(range(1, 13)), index=0)
with col2:
    meridian = st.selectbox("AM or PM", ["AM", "PM"])

# Convert hour + AM/PM into 24-hour time (integer)
if meridian == "PM" and hour != 12:
    time = hour + 12
elif meridian == "AM" and hour == 12:
    time = 0
else:
    time = hour

# Prediction
if st.button("Check Transaction"):
    try:
        # Create 30-feature array: [time, amount, 28 dummy values]
        dummy_values = [0.0] * 28
        input_features = [time, amount] + dummy_values  # Total: 30 features

        # Scale and predict
        input_scaled = scaler.transform([input_features])
        prediction = model.predict(input_scaled)

        # Output result
        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
