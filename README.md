# 🛒 Retail Pricing Strategy Optimization

An end-to-end AI/ML  project to predict optimal product prices, analyze price elasticity, and simulate pricing strategies through an interactive Streamlit dashboard.

---

## 🎯 Objective

The goal of this project is to optimize retail product pricing using historical sales data and machine learning. The project focuses on building a data-driven pricing model that predicts the impact of pricing decisions on sales volume and revenue, empowering businesses to adopt intelligent, adaptive pricing strategies.

---

## 📊 Dataset Description

The dataset includes the following key columns:
- `product_id`, `product_category_name`
- `unit_price`, `total_price`, `volume`, `qty`, `freight_price`
- `month_year`, `weekday`, `holiday`, `comp_1` (competitor price)
- Product attributes: length, weight, score
- Customer behavior across different time frames

---

## ⚙️ Tech Stack

- **Python** (Jupyter Notebook + .py scripts)
- **Pandas, NumPy, Scikit-learn**
- **Matplotlib, Seaborn** for visualization
- **Streamlit** for web app interface
- **Joblib** for model serialization

---

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing
- Handle missing values and outliers
- Encode categorical features (Label Encoding)
- Convert date strings to datetime
- Normalize numerical features

### 2. Exploratory Data Analysis (EDA)
- Visualize pricing and quantity trends
- Correlation heatmaps
- Category-wise analysis

### 3. Model Development
- Models used:
  - Linear Regression
  - Random Forest Regressor
- Target: `unit_price`
- Features: sales volume, weight, score, category, etc.
- Evaluation metrics: MSE, RMSE, R²

### 4. Price Elasticity Analysis
- Elasticity = %Δ Quantity / %Δ Price
- Categories marked as elastic, inelastic, or unitary
- Business insight: How price changes impact demand

### 5. Deployment
- Built Streamlit dashboard for price simulation
- Visualize trends and test new pricing scenarios
- Download prediction as CSV

---

## 🌐 Streamlit App

### 🔗 [Live Web App](https://retail-pricing-strategy-optimization.streamlit.app/)  

#### Features:
- Select product category
- Adjust price via slider
- Predict quantity sold and revenue
- Visualize historical price vs. quantity trend
- Download predictions

---

## 📁 Project Structure
```
retail-pricing-optimization/
├── retail_pricing_optimization.ipynb        # Jupyter Notebook: model training, EDA
├── streamlit_app.py                         # Streamlit UI for simulation
├── retail_price.csv                         # Source dataset
├── price_prediction_model.pkl               # Trained model file
├── requirements.txt                         # Project dependencies
├── Retail_Pricing_Final_Report.pdf          # PDF Report with screenshots
├── README.md                                # This documentation file
```
