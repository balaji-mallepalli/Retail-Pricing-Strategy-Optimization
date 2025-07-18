import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and expected columns
model_data = joblib.load("price_prediction_model.pkl")
model = model_data["model"]
expected_columns = model_data["columns"]

# Streamlit UI – Sidebar
st.sidebar.title("🛠️ Price Prediction Tool")

# Input fields
st.sidebar.markdown("### 📥 Enter Product Details")

freight_price = st.sidebar.number_input("Freight Price (₹)", min_value=0.0, value=20.0)
volume = st.sidebar.number_input("Volume (cm³)", min_value=0.0, value=500.0)
product_weight_g = st.sidebar.number_input("Product Weight (g)", min_value=0.0, value=600.0)
product_category = st.sidebar.number_input("Product Category (Encoded)", min_value=0, value=5)
product_score = st.sidebar.slider("Product Score", 0.0, 5.0, 4.2)
is_weekend = st.sidebar.radio("Is Weekend?", [0, 1])

# Create input dataframe
input_data = pd.DataFrame([[
    freight_price,
    volume,
    product_weight_g,
    product_category,
    product_score,
    is_weekend
]], columns=expected_columns)

# Prediction
st.title("💡 Unit Price Prediction")
st.markdown("This tool predicts the optimal **unit price** based on product features and context.")

try:
    predicted_price = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Unit Price: ₹{predicted_price:.2f}")
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("*Built by Balaji Mallepalli – SRM University AP || AI&ML Project*")
