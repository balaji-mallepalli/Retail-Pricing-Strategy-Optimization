import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- MODEL & DATA LOADING ---

# Load trained model (pipeline) and column order
try:
    model_data = joblib.load("price_prediction_model.pkl")
    model = model_data["model"]
    expected_columns = model_data.get("columns", None)
except Exception:
    st.error("❌ Could not load the ML model. Check filename or path.")
    st.stop()
# Load dataset
try:
    df = pd.read_csv("retail_price.csv")
except Exception:
    st.error("❌ Could not load dataset. Make sure 'retail_price.csv' is present.")
    st.stop()

# Convert to datetime if present
if 'month_year' in df.columns:
    try:
        df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')
    except Exception:
        st.warning("⚠️ 'month_year' could not be fully parsed as datetime.")

# --- SIDEBAR INPUTS ---

st.sidebar.title("🔧 Pricing Simulator")

# Category selection
if 'product_category_name' not in df.columns:
    st.error("'product_category_name' column missing.")
    st.stop()
df['product_category_name'] = df['product_category_name'].astype(str)
categories = sorted(df['product_category_name'].unique())
category = st.sidebar.selectbox("Select Product Category", categories)

# Filter by chosen category
filtered_df = df[df['product_category_name'] == category]
if filtered_df.empty:
    st.error("❌ No products found for selected category.")
    st.stop()

# Product selection within category
products = sorted(filtered_df['product_id'].unique())
product_id = st.sidebar.selectbox("Select Product", products)
prod_df = filtered_df[filtered_df['product_id'] == product_id]
if prod_df.empty:
    st.error("❌ No data for selected product.")
    st.stop()

# Price slider for simulation
price_min = float(prod_df['unit_price'].min())
price_max = float(prod_df['unit_price'].max())
price_default = float(prod_df['unit_price'].mean())

price_input = st.sidebar.slider("Set New Price", price_min, price_max, price_default, step=0.1)

# --- MAIN PANEL ---

st.title("💰 Retail Pricing Strategy Optimization")
st.markdown(f"### Simulating: **{category}** / **{product_id}** at ₹{price_input:.2f}")

# Prepare input for model
# Use mean values for all non-price features of this product (except price)
# If you did OHE in pipeline, this works; if you did manual encoding, ensure all dummies included
input_row = prod_df.mean(numeric_only=True).to_frame().T
if 'unit_price' in input_row.columns:
    input_row['unit_price'] = price_input
if 'price_vs_comp' in input_row.columns and 'comp_avg_price' in input_row.columns:
    input_row['price_vs_comp'] = price_input - input_row['comp_avg_price']

# Prediction
try:
    pred = model.predict(input_row)[0]
    # If your model uses log-transformed qty, inverse it
    if 'log_qty' in prod_df.columns:
        pred_qty = np.expm1(pred)
    else:
        pred_qty = pred
    revenue = pred_qty * price_input
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# Results
st.metric("📦 Predicted Quantity Sold", f"{pred_qty:,.0f} units")
st.metric("💵 Estimated Revenue", f"₹{revenue:,.2f}")

# --- PRICE VS QUANTITY PLOT ---

st.subheader("📈 Historical Price vs Quantity (Selected Product)")
fig, ax = plt.subplots()
ax.scatter(prod_df['unit_price'], prod_df['qty'], alpha=0.6, label='Historical')
ax.axvline(price_input, color='red', linestyle='--', label='Your Simulation')
ax.set_xlabel("Unit Price (₹)")
ax.set_ylabel("Quantity Sold")
ax.legend()
st.pyplot(fig)

# --- (Optional) Simulate Revenue Curve Over Price Grid ---

st.subheader("🧮 Simulated Revenue vs Price")
price_grid = np.linspace(price_min, price_max, 40)
input_grid = pd.concat([input_row]*len(price_grid), ignore_index=True)
input_grid['unit_price'] = price_grid
if 'price_vs_comp' in input_grid.columns and 'comp_avg_price' in input_grid.columns:
    input_grid['price_vs_comp'] = price_grid - input_grid['comp_avg_price']

try:
    pred_grid = model.predict(input_grid)
    if 'log_qty' in prod_df.columns:
        pred_qties = np.expm1(pred_grid)
    else:
        pred_qties = pred_grid
    revs = price_grid * pred_qties
    plt.figure(figsize=(6,4))
    plt.plot(price_grid, revs)
    plt.xlabel("Unit Price (₹)")
    plt.ylabel("Predicted Revenue")
    plt.axvline(price_input, color='red', linestyle='--', label='Selected Price')
    plt.title(f"Revenue vs Price for {product_id}")
    plt.legend()
    st.pyplot(plt)
except Exception as e:
    st.warning(f"Simulation plot failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("*Built by Balaji Mallepalli – SRM University AP || AI/ML  Project*")
