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
# TABEL INDIKATOR REFERENSI FAKTUAL (STANDAR MEDIS WHO/ADA & DATASET)
# =====================================================================
with st.expander("Lihat Tabel Indikator Referensi Medis Faktual"):
    st.write("Rentang nilai di bawah ini menggunakan Standar Medis Internasional (WHO/ADA) dan nilai rata-rata riil dari dataset:")
    
    data_indikator = {
        "Fitur Medis": [
            "Glucose (Glukosa Darah 2 Jam Pasca Makan)",
            "Blood Pressure (Tekanan Darah Diastolik)",
            "BMI (Indeks Massa Tubuh)",
            "Pregnancies (Jumlah Kehamilan)",
            "Skin Thickness (Ketebalan Kulit Trisep)",
            "Insulin (Kadar Insulin 2 Jam)",
            "Diabetes Pedigree Function (Skor Genetik)",
            "Age (Umur)"
        ],
        "Normal / Rentang Sehat": [
            "< 140 mg/dL (Standar ADA)",
            "< 80 mmHg (Standar AHA/ADA)",
            "18.5 - 24.9 (Standar WHO)",
            "0 - 3 kali (Mayoritas Data Sehat)",
            "10 - 29 mm (Rata-rata Normal)",
            "< 160 mIU/L (Rentang Normal)",
            "< 0.500 (Riwayat Keluarga Rendah)",
            "21 - 30 tahun"
        ],
        "Kriteria Risiko / Diabetes": [
            ">= 200 mg/dL (Diabetes) | 140-199 (Pre-Diabetes)",
            ">= 80 mmHg (Hipertensi Tahap 1 & 2)",
            ">= 30.0 (Obesitas) | 25.0-29.9 (Overweight)",
            "> 4 kali (Pola Risiko Dataset)",
            ">= 33 mm (Pola Tinggi Pasien Diabetes)",
            "> 160 mIU/L (Hiperinsulinemia)",
            ">= 0.500 (Riwayat Keluarga Tinggi)",
            "> 30 tahun (Penyebaran Kasus di Dataset)"
        ]
    }
    
    df_indikator = pd.DataFrame(data_indikator)
    st.table(df_indikator)
    st.caption("Sumber Referensi: American Diabetes Association (ADA), World Health Organization (WHO), dan Distribusi Statistik Dataset Pima Indians.")

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
