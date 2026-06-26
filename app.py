import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load file pkl yang berisi bundle model dan scaler
with open('diabetes_prediction.pkl', 'rb') as f:
    model_bundle = pickle.load(f)

# Pisahkan objek model dan scaler dari dictionary
model = model_bundle["model"]
scaler = model_bundle["scaler"]

st.title("Aplikasi Prediksi Diabetes - Algoritma Extra Trees")
st.write("Masukkan data medis Anda di bawah ini untuk melihat hasil prediksi.")

# =====================================================================
# TABEL INDIKATOR REFERENSI (DATASET PIMA INDIANS DIABETES)
# =====================================================================
with st.expander("Lihat Tabel Indikator Referensi Medis"):
    st.write("Rentang nilai ini merupakan acuan umum berdasarkan pola data pasien dalam dataset:")
    
    data_indikator = {
        "Fitur Medis": [
            "Pregnancies (Kehamilan)",
            "Glucose (Glukosa Darah)",
            "Blood Pressure (Tekanan Darah)",
            "Skin Thickness (Ketebalan Kulit)",
            "Insulin",
            "BMI (Indeks Massa Tubuh)",
            "Diabetes Pedigree Function (DPF)",
            "Age (Umur)"
        ],
        "Cenderung Sehat (0)": [
            "0 - 3 kali",
            "< 120 mg/dL",
            "60 - 80 mmHg",
            "< 20 mm",
            "< 100 mIU/L",
            "18.5 - 25.0",
            "< 0.400",
            "21 - 30 tahun"
        ],
        "Risiko Diabetes (1)": [
            "> 5 kali",
            "> 130 mg/dL (Sangat Sensitif)",
            "> 85 mmHg",
            "> 30 mm",
            "> 150 mIU/L",
            "> 30.0 (Obesitas)",
            "> 0.500 (Riwayat Kuat)",
            "> 35 tahun"
        ]
    }
    
    df_indikator = pd.DataFrame(data_indikator)
    st.table(df_indikator)
    st.caption("Catatan: AI menentukan hasil berdasarkan kombinasi seluruh fitur di atas secara bersamaan, bukan dari satu fitur saja.")

# =====================================================================
# FORM INPUT DATA MEDIS
# =====================================================================
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

if submit:
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Transformasi data baru menggunakan scaler bawaan
    features_scaled = scaler.transform(features)
    
    # Prediksi menggunakan data yang sudah di-scale
    prediction = model.predict(features_scaled)[0]
    
    st.write("---")
    if prediction == 1:
        st.error("Hasil Analisis: Positif Diabetes")
    else:
        st.success("Hasil Analisis: Tidak Terindikasi Diabetes")
