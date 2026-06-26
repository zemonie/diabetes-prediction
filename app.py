import streamlit as st
import numpy as np
import pickle

# 1. Load model Extra Trees milikmu yang sudah di-rename
with open('diabetes_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

# Judul Website
st.title("Aplikasi Prediksi Diabetes - Algoritma Extra Trees")
st.write("Masukkan data medis Anda di bawah ini untuk melihat hasil prediksi.")

# Form Input Data Medis
with st.form("form_diabetes_kamu"):
    pregnancies = st.number_input('Pregnancies (Jumlah Kehamilan)', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose (Kadar Glukosa)', min_value=0, max_value=200)
    blood_pressure = st.number_input('Blood Pressure (Tekanan Darah)', min_value=0, max_value=150)
    skin_thickness = st.number_input('Skin Thickness (Ketebalan Kulit)', min_value=0, max_value=100)
    insulin = st.number_input('Insulin', min_value=0, max_value=1000)
    bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=0.0, max_value=70.0, format="%.1f")
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, format="%.3f")
    age = st.number_input('Age (Umur)', min_value=1, max_value=120, step=1)
    
    submit = st.form_submit_button("Proses")

# Logika Output ketika tombol Proses ditekan
if submit:
    # Format ke bentuk array 2D
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Prediksi menggunakan model Extra Trees milikmu
    prediction = model.predict(features)[0]
    
    st.write("---")
    if prediction == 1:
        st.error("Hasil Analisis: Positif Diabetes")
    else:
        st.success("Hasil Analisis: Tidak Terindikasi Diabetes")
