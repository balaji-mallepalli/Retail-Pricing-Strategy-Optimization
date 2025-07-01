import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load Model and Feature Names
model_data = joblib.load("price_prediction_model.pkl")
model = model_data["model"]
expected_columns = model_data["columns"]

# Load Dataset
df = pd.read_csv("retail_price.csv")

# Safe date conversion
if 'month_year' in df.columns:
    try:
        df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')
    except Exception:
        st.warning("⚠️ Could not convert 'month_year' column to datetime.")

# Encode product category for selection
if 'product_category_name' in df.columns:
    df['product_category_name'] = df['product_category_name'].astype(str)
    category_list = df['product_category_name'].unique()
else:
    st.error("❌ 'product_category_name' column not found in dataset.")
    st.stop()

# Sidebar Inputs
st.sidebar.title("🔧 Pricing Simulator")
selected_category = st.sidebar.selectbox("Select Product Category", sorted(category_list))

filtered_df = df[df['product_category_name'] == selected_category]

# Check if enough data exists
if filtered_df.shape[0] < 1:
    st.error("No data available for the selected category.")
    st.stop()

price_range = (filtered_df['unit_price'].min(), filtered_df['unit_price'].max())
default_price = filtered_df['unit_price'].mean()
price_input = st.sidebar.slider("Set New Price", float(price_range[0]), float(price_range[1]), float(default_price))


# Title & Intro
st.title("💰 Retail Pricing Strategy Optimization")
st.markdown(f"### Simulating for **{selected_category}** at ₹{price_input:.2f}")

# Prepare Input for Prediction
# Compute mean row for selected category
user_input = filtered_df[expected_columns].mean().to_frame().T

# Replace price
if 'unit_price' in user_input.columns:
    user_input['unit_price'] = price_input

# Ensure same column order and type
user_input = user_input[expected_columns].fillna(0)

# Make Prediction
try:
    predicted_quantity = model.predict(user_input)[0]
    estimated_revenue = predicted_quantity * price_input
except Exception as e:
    st.error(f"❌ Prediction failed: {str(e)}")
    st.stop()

# Output Results
st.metric("📦 Predicted Quantity Sold", f"{predicted_quantity:,.0f} units")
st.metric("💵 Estimated Revenue", f"₹{estimated_revenue:,.2f}")

# Plot: Price vs Quantity (Historical)
st.subheader("📈 Historical Price vs Quantity Trend")
fig, ax = plt.subplots()
ax.scatter(filtered_df['unit_price'], filtered_df['qty'], alpha=0.6, label='Historical')
ax.axvline(price_input, color='red', linestyle='--', label='Selected Price')
ax.set_xlabel("Unit Price (₹)")
ax.set_ylabel("Quantity Sold")
ax.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🔍 *Built by Balaji Mallepalli – SRM University | AI/ML Capstone Project*")
