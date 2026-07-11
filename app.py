import streamlit as st
import numpy as np
import pickle
import pandas as pd

# 1. LOAD MODEL & SCALER (Memuat bundle hasil training yang sudah disimpan)
with open('diabetes_prediction.pkl', 'rb') as f:
    model_bundle = pickle.load(f)

# Memisahkan objek agar proses transformasi dan prediksi berjalan sinkron
model = model_bundle["model"]
scaler = model_bundle["scaler"]

# 2. ANTARMUKA (UI) UTAMA APLIKASI
st.title("Aplikasi Prediksi Diabetes - Algoritma Extra Trees")
st.write("Masukkan data medis Anda di bawah ini untuk melihat hasil prediksi.")

# =====================================================================
# TABEL INDIKATOR REFERENSI FAKTUAL (URUTAN SESUAI FORM INPUT)
# =====================================================================
# Menampilkan expander referensi medis sebagai dasar validasi ilmiah aplikasi
with st.expander("Lihat Tabel Indikator Referensi Medis Faktual"):
    st.write("Rentang nilai di bawah ini menggunakan Standar Medis Internasional (WHO/ADA) dan nilai rata-rata riil dari dataset:")
    
    data_indikator = {
        "Fitur Medis": [
            "Pregnancies (Jumlah Kehamilan)",
            "Glucose (Glukosa Darah 2 Jam Pasca Makan)",
            "Blood Pressure (Tekanan Darah Diastolik)",
            "Skin Thickness (Ketebalan Kulit Trisep)",
            "Insulin (Kadar Insulin 2 Jam)",
            "BMI (Indeks Massa Tubuh)",
            "Diabetes Pedigree Function (Skor Genetik)",
            "Age (Umur)"
        ],
        "Normal / Rentang Sehat": [
            "0 - 3 kali (Mayoritas Data Sehat)",
            "< 140 mg/dL (Standar ADA)",
            "< 80 mmHg (Standar AHA/ADA)",
            "10 - 29 mm (Rata-rata Normal)",
            "< 160 mIU/L (Rentang Normal)",
            "18.5 - 24.9 (Standar WHO)",
            "< 0.500 (Riwayat Keluarga Rendah)",
            "21 - 30 tahun"
        ],
        "Kriteria Risiko / Diabetes": [
            "Obat > 4 kali (Pola Risiko Dataset)",
            ">= 200 mg/dL (Diabetes) | 140-199 (Pre-Diabetes)",
            ">= 80 mmHg (Hipertensi Tahap 1 & 2)",
            ">= 33 mm (Pola Tinggi Pasien Diabetes)",
            "> 160 mIU/L (Hiperinsulinemia)",
            ">= 30.0 (Obesitas) | 25.0-29.9 (Overweight)",
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
# Menyediakan form input interaktif untuk menangkap data pasien baru
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

# =====================================================================
# PROSES PREDIKSI (DIJALANKAN SAAT TOMBOL PROSES DIKLIK)
# =====================================================================
if submit:
    # Menggabungkan seluruh input form menjadi satu array 2D
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # WAJIB: Menyamakan skala data baru menggunakan scaler yang dilatih saat training (StandardScaler)
    features_scaled = scaler.transform(features)
    
    # Melakukan klasifikasi menggunakan data yang sudah disetarakan skalanya
    prediction = model.predict(features_scaled)[0]
    
    # TAMPILAN OUTPUT PREDIKSI
    st.write("---")
    if prediction == 1:
        st.error("Hasil Analisis: Positif Diabetes")
    else:
        st.success("Hasil Analisis: Tidak Terindikasi Diabetes")
