import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load Model and Data
model = joblib.load("price_prediction_model.pkl")
df = pd.read_csv("retail_price.csv")

# Optional: convert date
# Fix for date conversion crash
if 'month_year' in df.columns:
    try:
        df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')
    except Exception as e:
        st.warning("⚠️ Could not convert 'month_year' column to datetime.")


# Encoding category
df['product_category_name_encoded'] = df['product_category_name'].astype('category').cat.codes
category_map = dict(enumerate(df['product_category_name'].astype('category').cat.categories))

# Sidebar – Inputs
st.sidebar.title("🔧 Pricing Simulator")
category = st.sidebar.selectbox("Select Product Category", df['product_category_name'].unique())
filtered_df = df[df['product_category_name'] == category]

price_input = st.sidebar.slider("Set New Price", 
                                float(filtered_df['unit_price'].min()), 
                                float(filtered_df['unit_price'].max()), 
                                float(filtered_df['unit_price'].mean()))

# Main Panel
st.title("💰 Retail Pricing Strategy Optimization")
st.markdown(f"### Simulating for **{category}** at ₹{price_input:.2f}")

# Prepare input
X_features = filtered_df.drop(columns=['unit_price', 'product_id', 'product_category_name'], errors='ignore').mean().to_frame().T

# Replace with new price
if 'unit_price' in X_features.columns:
    X_features['unit_price'] = price_input

# Predict
predicted_quantity = model.predict(X_features)[0]
estimated_revenue = predicted_quantity * price_input

# Output
st.metric("📦 Predicted Quantity Sold", f"{predicted_quantity:.0f} units")
st.metric("💵 Estimated Revenue", f"₹{estimated_revenue:,.2f}")

# Elasticity Visualization
st.subheader("📈 Historical Price vs Quantity Trend")
fig, ax = plt.subplots()
ax.plot(filtered_df['unit_price'], filtered_df['qty'], marker='o')
ax.set_xlabel("Unit Price")
ax.set_ylabel("Quantity Sold")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built by Balaji Mallepalli | SRM University AP")
