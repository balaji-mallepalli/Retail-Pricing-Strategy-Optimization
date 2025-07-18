
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and expected columns
model_data = joblib.load("price_prediction_model.pkl")
model = model_data["model"]
expected_columns = model_data["columns"]

# Load dataset
df = pd.read_csv("retail_price.csv")

# Convert to datetime safely
if 'month_year' in df.columns:
    try:
        df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')
    except Exception:
        st.warning("⚠️ Could not convert 'month_year' to datetime")

# Convert product_category to string and encode
if 'product_category_name' in df.columns:
    df['product_category_name'] = df['product_category_name'].astype(str)
    df['product_category_name_encoded'] = df['product_category_name'].astype('category').cat.codes
else:
    st.error("❌ 'product_category_name' column is missing in the dataset.")
    st.stop()

# Streamlit UI – Sidebar
st.sidebar.title("🔧 Pricing Simulator")

category_list = sorted(df['product_category_name'].unique())
selected_category = st.sidebar.selectbox("Select Product Category", category_list)

filtered_df = df[df['product_category_name'] == selected_category]

if filtered_df.empty:
    st.error("❌ No data available for the selected category.")
    st.stop()

# Slider for setting price
price_min = float(filtered_df['unit_price'].min())
price_max = float(filtered_df['unit_price'].max())
default_price = float(filtered_df['unit_price'].mean())

price_input = st.sidebar.slider("Set New Price", price_min, price_max, default_price)

# Main Panel
st.title("💰 Retail Pricing Strategy Optimization")
st.markdown(f"### Simulating for **{selected_category}** at ₹{price_input:.2f}")

# Prepare prediction input
# Ensure all expected columns are present
for col in expected_columns:
    if col not in filtered_df.columns:
        filtered_df[col] = 0  # Fill missing with default 0

# Create input row
user_input = filtered_df[expected_columns].mean().to_frame().T
if 'unit_price' in user_input.columns:
    user_input['unit_price'] = price_input

# Predict
try:
    predicted_quantity = model.predict(user_input)[0]
    estimated_revenue = predicted_quantity * price_input
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# Show results
st.metric("📦 Predicted Quantity Sold", f"{predicted_quantity:,.0f} units")
st.metric("💵 Estimated Revenue", f"₹{estimated_revenue:,.2f}")

# Price vs Quantity Chart
st.subheader("📈 Historical Price vs Quantity")

fig, ax = plt.subplots()
ax.scatter(filtered_df['unit_price'], filtered_df['qty'], alpha=0.6, label='Historical')
ax.axvline(price_input, color='red', linestyle='--', label='Selected Price')
ax.set_xlabel("Unit Price (₹)")
ax.set_ylabel("Quantity Sold")
ax.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("*Built by Balaji Mallepalli – SRM University AP || AI&ML Project*")
