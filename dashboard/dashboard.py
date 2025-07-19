import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Config
st.set_page_config(page_title="Dashboard Prediksi Harga HP", layout="wide")
st.title("📱 Dashboard Analisis dan Prediksi Harga Handphone")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("dataset_handphone_cleaned.csv") 

df = load_data()

# Sidebar Menu
menu = st.sidebar.radio("Navigasi", ["📊 EDA", "📈 Korelasi", "🤖 Prediksi Harga", "📋 Evaluasi Model"])

# --- EDA ---
if menu == "📊 EDA":
    st.header("Exploratory Data Analysis")
    st.dataframe(df.head())

    st.subheader("Distribusi Harga")
    fig, ax = plt.subplots()
    sns.histplot(df['Harga'], bins=30, kde=True, ax=ax, color='orange')
    st.pyplot(fig)

    st.subheader("RAM vs Harga")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='Ram', y='Harga', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Internal Memory vs Harga")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Memori_internal', y='Harga', data=df, ax=ax3)
    st.pyplot(fig3)

# --- Korelasi ---
elif menu == "📈 Korelasi":
    st.header("Korelasi Fitur")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- Prediksi Harga ---
elif menu == "🤖 Prediksi Harga":
    st.header("Prediksi Harga HP")

    ram = st.number_input("RAM (GB)", 1, 32, 4)
    internal = st.number_input("Internal Memory (GB)", 8, 512, 64)
    battery = st.number_input("Baterai (mAh)", 1000, 7000, 4000)
    screen_size = st.number_input("Ukuran Layar (inci)", 4.0, 7.0, 6.5)

    X = df[['Ram', 'Memori_internal', 'Kapasitas_baterai', 'Ukuran_layar']]
    y = df['Harga']

    model = LinearRegression().fit(X, y)
    pred = model.predict([[ram, internal, battery, screen_size]])[0]

    st.success(f"Prediksi Harga: Rp {pred:,.0f}")

# --- Evaluasi Model ---
elif menu == "📋 Evaluasi Model":
    st.header("Evaluasi Model Regresi")
    X = df[['Ram', 'Memori_internal', 'Kapasitas_baterai', 'Ukuran_layar']]
    y = df['Harga']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Hasil Evaluasi")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    st.subheader("Plot Prediksi vs Aktual")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    ax.set_xlabel("Harga Aktual")
    ax.set_ylabel("Harga Prediksi")
    st.pyplot(fig)
