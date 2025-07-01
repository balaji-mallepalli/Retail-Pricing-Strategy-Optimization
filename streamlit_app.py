import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and expected features
model_data = joblib.load("price_prediction_model.pkl")
model = model_data["model"]
expected_columns = model_data["columns"]

# Load dataset
df = pd.read_csv("retail_price.csv")

# Safe datetime conversion
if 'month_year' in df.columns:
    try:
        df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')
    except Exception:
        st.warning("⚠️ Could not convert 'month_year' to datetime.")

# Ensure product_category_name exists
if 'product_category_name' not in df.columns:
    st.error("❌ 'product_category_name' column is missing.")
    st.stop()

df['product_category_name'] = df['product_category_name'].astype(str)
df['product_category_name_encoded'] = df['product_category_name'].astype('category').cat.codes

# Sidebar Inputs
st.sidebar.title("🔧 Pricing Simulator")
category_list = sorted(df['product_category_name'].unique())
selected_category = st.sidebar.selectbox("Select Product Category", category_list)

filtered_df = df[df['product_category_name'] == selected_category]

if filtered_df.empty:
    st.error("❌ No data available for the selected category.")
    st.stop()

# Price input slider
min_price = float(filtered_df['unit_price'].min())
max_price = float(filtered_df['unit_price'].max())
default_price = float(filtered_df['unit_price'].mean())

price_input = st.sidebar.slider("Set New Price", min_value=min_price, max_value=max_price, value=default_price)

# Main UI
st.title("💰 Retail Pricing Strategy Optimization")
st.markdown(f"### Simulating for **{selected_category}** at ₹{price_input:.2f}")

# Prepare Input for Prediction

# Fill any missing expected columns
for col in expected_columns:
    if col not in filtered_df.columns:
        filtered_df[col] = 0

# Prepare mean input row
user_input = filtered_df[expected_columns].mean().to_frame().T

# Update the selected price
if 'unit_price' in user_input.columns:
    user_input['unit_price'] = price_input

# Prediction
try:
    predicted_quantity = model.predict(user_input)[0]
    estimated_revenue = predicted_quantity * price_input
except Exception as e:
    st.error(f"❌ Prediction failed: {str(e)}")
    st.stop()

# Display Metrics
st.metric("📦 Predicted Quantity Sold", f"{predicted_quantity:,.0f} units")
st.metric("💵 Estimated Revenue", f"₹{estimated_revenue:,.2f}")

# 📥 Download Button for Prediction Input
csv_data = user_input.copy()
csv_data["Predicted Quantity"] = predicted_quantity
csv_data["Estimated Revenue"] = estimated_revenue

st.download_button(
    label="📥 Download Prediction CSV",
    data=csv_data.to_csv(index=False),
    file_name="prediction_output.csv",
    mime="text/csv"
)

# Visualization
st.subheader("📈 Historical Price vs Quantity Trend")
fig, ax = plt.subplots()
ax.scatter(filtered_df['unit_price'], filtered_df['qty'], alpha=0.6, label='Historical')
ax.axvline(price_input, color='red', linestyle='--', label='Selected Price')
ax.set_xlabel("Unit Price (₹)")
ax.set_ylabel("Quantity Sold")
ax.set_title(f"{selected_category} — Price vs Quantity")
ax.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("✅ Built by **Balaji Mallepalli** – SRM University AP || AI&ML  Project 2025")
