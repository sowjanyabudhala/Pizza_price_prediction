# 🍕 Pizza Price Prediction using Gradient Boosting Regressor

## 📌 Table of Contents
- [Demo](#demo)
- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [Run](#run)
- [Deployment (Streamlit)](#deployment-streamlit)
- [Directory Structure](#directory-structure)
- [To-Do](#to-do)
- [Bug / Feature Request](#bug--feature-request)
- [Technologies Used](#technologies-used)
- [Credits](#credits)

---

## 🎥 Demo
🔗 **Live Demo**: [Add your deployment link here]  

![Demo](https://your-demo-link.com/demo.gif)

---

## 📖 Overview
This project predicts **pizza prices** based on factors like **diameter, company, toppings, size, and extra ingredients** using machine learning techniques, specifically the **Gradient Boosting Regressor**. The dataset includes various pizza features and their corresponding prices.

💡 **Models Used**:
- Linear Regression
- Support Vector Machine (SVR)
- Random Forest Regressor
- Gradient Boosting Regressor (**Best Performing Model**)
- XGBoost Regressor

📊 **Evaluation Metrics**:
- R² Score

---

## 🎯 Motivation
Understanding **pizza pricing** is essential for both **business owners** and **consumers**. This project helps in:
- Predicting **optimal pizza prices** based on various features.
- Analyzing the **impact of size, toppings, and extras** on pricing.
- Building a **data-driven pricing strategy** for pizza businesses.

---

## ⚙️ Technical Aspect
This project consists of **two main components**:
1. **Model Training & Evaluation**:
   - **Data Preprocessing**: Handling missing values, feature conversion, and encoding categorical data.
   - **Feature Engineering**: Converting currency values, removing outliers, and extracting relevant features.
   - **Model Training**: Comparing multiple regression models.
   - **Model Evaluation**: Using R² Score for performance comparison.
   - **Best Model Selection**: Gradient Boosting Regressor is used for final predictions.

2. **Model Deployment using Streamlit**:
   - A **user-friendly UI** where users can input pizza details.
   - Predicts **pizza price** based on input factors.

---

## 💻 Installation
Ensure that **Python 3.7 or above** is installed: [Download Python](https://www.python.org/downloads/).  

To install dependencies, run:
```bash
pip install -r requirements.txt

---



##🚀 Run
Step 1: Setting Environment Variables
python pizza_price_prediction_by_gradient_boosting_regressor.py

🌐 Deployment (Streamlit)
To deploy this model using Streamlit, follow these steps:

Step 1: Install Streamlit
pip install streamlit
Step 2: Run the App
streamlit run app.py



📂 Directory Structure
📦 Pizza Price Prediction
 ┣ 📂 data
 ┃ ┣ 📄 pizza_v2.csv
 ┣ 📂 models
 ┃ ┣ 📄 pizza_price_predict.pkl
 ┣ 📂 scripts
 ┃ ┣ 📄 pizza_price_prediction_by_gradient_boosting_regressor.py
 ┣ 📂 streamlit_app
 ┃ ┣ 📄 app.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┣ 📄 LICENSE


✅ To-Do
✔ Deploy the model using Flask API
✔ Enable real-time pizza price estimation

🐞 Bug / Feature Request
If you find a bug or want to request a new feature, open an issue:
📌 GitHub Issues

🛠 Technologies Used
### 1️⃣ Scikit-learn
- Terminology: Machine Learning Library  
- Description: Provides tools for data preprocessing, classification, regression, clustering, and model evaluation.  
- Usage: Used for train-test splitting, model training, and evaluation metrics (R² Score).

## 2️⃣ Streamlit 
- Terminology: Web App Framework for Machine Learning Models  
- Description: Enables real-time user interaction and model deployment through a web UI.  
- Usage: Used for deploying the calorie burn prediction model.

## 3️⃣ Pandas 
- Terminology: Data Analysis & Manipulation Library  
- Description: Helps in handling structured datasets (CSV, Excel) and performing data preprocessing.  
- Usage: Used for loading, cleaning, and transforming exercise and calorie data.  

### 4️⃣ Matplotlib 
- Terminology: Data Visualization Library  
- Description: Used for generating graphs and charts to analyze trends in calorie burn.  
- Usage: Used for histograms, scatter plots, and heatmaps.

🙌 Credits
Dataset: Kaggle Pizza Price Dataset
Contributors: B. Sowjanya

   
