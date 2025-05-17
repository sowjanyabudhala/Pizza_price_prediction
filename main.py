import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("pizza_v2.csv")
    
    # Data Preprocessing
    df.rename(columns={'price_rupiah': 'Price'}, inplace=True)
    df['Price'] = df['Price'].str.replace('Rp', '').str.replace(',', '').astype(float)
    df['Price'] *= 0.0054  # Convert currency

    df['diameter'] = df['diameter'].str.replace(' inch', '').astype(float)

    category_cols = ['company', 'topping', 'variant', 'size', 'extra_sauce', 'extra_cheese', 'extra_mushrooms']
    le = LabelEncoder()
    for col in category_cols:
        df[col] = le.fit_transform(df[col])

    return df

df = load_data()

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Splitting features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Model Evaluation
y_pred = gbr.predict(X_test)
r2_score = metrics.r2_score(y_test, y_pred)
st.write(f"Model R² Score: {r2_score:.2f}")

# Sidebar Input Form
st.sidebar.header("Enter Pizza Details")

company = st.sidebar.selectbox("Company", df['company'].unique())
diameter = st.sidebar.number_input("Diameter (inches)", min_value=6.0, max_value=30.0)
topping = st.sidebar.selectbox("Topping", df['topping'].unique())
variant = st.sidebar.selectbox("Variant", df['variant'].unique())
size = st.sidebar.selectbox("Size", df['size'].unique())
extra_sauce = st.sidebar.selectbox("Extra Sauce", df['extra_sauce'].unique())
extra_cheese = st.sidebar.selectbox("Extra Cheese", df['extra_cheese'].unique())
extra_mushrooms = st.sidebar.selectbox("Extra Mushrooms", df['extra_mushrooms'].unique())

# Predict Button
if st.sidebar.button("Predict Price"):
    user_data = pd.DataFrame([[company, diameter, topping, variant, size, extra_sauce, extra_cheese, extra_mushrooms]],
                             columns=['company', 'diameter', 'topping', 'variant', 'size', 'extra_sauce', 'extra_cheese', 'extra_mushrooms'])
    prediction = gbr.predict(user_data)
    st.sidebar.success(f"Estimated Pizza Price: ${prediction[0]:.2f}")

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)
