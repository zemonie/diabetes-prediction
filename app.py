import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Konfigurasi halaman diletakkan di paling atas agar tampilan melebar
st.set_page_config(layout="wide")

# 1. LOAD MODEL & SCALER STERIL
with open('diabetes_prediction.pkl', 'rb') as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
scaler = model_bundle["scaler"]

st.title("Aplikasi Prediksi Diabetes - Algoritma Random Forest")
st.write("Masukkan data medis Anda pada form di sebelah kiri untuk melihat hasil analisis prediksi.")
st.write("---")

# MEMBUAT TAMPILAN 2 KOLOM
col1, col2 = st.columns([4, 5], gap="large")

# ==================== KOLOM 1: FORM INPUT DATA ====================
with col1:
    st.subheader("Form Data Medis")
    
    with st.form("form_diabetes_kamu"):
        pregnancies = st.number_input('Pregnancies (Jumlah Kehamilan)', min_value=0, max_value=20, step=1, value=0)
        glucose = st.number_input('Glucose (Kadar Glukosa)', min_value=0, max_value=500, value=0) 
        blood_pressure = st.number_input('Blood Pressure (Tekanan Darah)', min_value=0, max_value=240, value=0) 
        skin_thickness = st.number_input('Skin Thickness (Ketebalan Kulit)', min_value=0, max_value=100, value=0)
        insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0) 
        bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=0.0, max_value=70.0, format="%.1f", value=0.0)
        dpf = st.number_input('Diabetes Pedigree Function (Skor Genetik)', min_value=0.000, max_value=3.000, format="%.3f", value=0.000)
        age = st.number_input('Age (Umur)', min_value=0, max_value=120, step=1, value=0)
        
        submit = st.form_submit_button("Proses Analisis Medis")

# ==================== KOLOM 2: TABEL INDIKATOR REFERENSI ====================
with col2:
    st.subheader("Panduan Indikator Medis Faktual")
    st.info("Gunakan tabel referensi di bawah ini sebagai panduan standar internasional (WHO/ADA) saat mengisi form:")
    
    # REVISI: Menggunakan struktur tabel Markdown agar tampilan visualnya jauh lebih rapi dan elegan
    st.markdown("""
    | Fitur Medis | Rentang Normal / Sehat | Rentang Risiko / Diabetes |
    | :--- | :--- | :--- |
    | **Pregnancies** (Jumlah Kehamilan) | 0 - 3 kali | ≥ 4 kali |
    | **Glucose** (Glukosa Darah 2 Jam Pasca Makan) | < 140 mg/dL | 140 - 199 mg/dL (Pre) \| ≥ 200 mg/dL (Diabetes) |
    | **Blood Pressure** (Tekanan Darah Diastolik) | < 80 mmHg | ≥ 80 mmHg (Hipertensi) |
    | **Skin Thickness** (Ketebalan Kulit Trisep) | 10 - 29 mm | ≥ 30 mm (Akumulasi Lemak) |
    | **Insulin** (Kadar Insulin 2 Jam) | < 160 mIU/L | ≥ 160 mIU/L (Resistensi Insulin) |
    | **BMI** (Indeks Massa Tubuh) | 18.5 - 24.9 | 25.0 - 29.9 (Overweight) \| ≥ 30.0 (Obesitas) |
    | **Diabetes Pedigree Function** (Skor Genetik) | < 0.500 | ≥ 0.500 (Riwayat Keluarga Kuat) |
    | **Age** (Umur) | 21 - 30 tahun | > 30 tahun (Metabolisme Menurun) |
    """)

# ==================== PROSES PREDIKSI & OUTPUT HASIL ====================
if submit:
    kolom_asli = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=kolom_asli)
    
    # Fitur Scaling secara steril menggunakan objek scaler bawaan dari pkl
    features_scaled = scaler.transform(input_df)
    
    # Prediksi Klasifikasi
    prediction = model.predict(features_scaled)[0]
    
    st.write("---")
    st.subheader("Hasil Analisis Model Sistem")
    
    if prediction == 1:
        st.error("Hasil Analisis Medis: Pasien Terindikasi POSITIF Diabetes Mellitus")
        st.markdown("""
        **Rekomendasi Akademis & Medis:** 
        * Segera lakukan pemeriksaan penunjang lanjutan (seperti tes HbA1c) dan konsultasi dengan dokter spesialis penyakit dalam (Endokrinolog).
        * Mulai batasi asupan karbohidrat sederhana dan makanan dengan indeks glikemik tinggi.
        """)
    else:
        st.success("Hasil Analisis Medis: Pasien NEGATIF / Tidak Terindikasi Diabetes Mellitus")
        st.markdown("""
        **Rekomendasi Akademis & Medis:** 
        * Pertahankan pola hidup sehat yang dijalankan saat ini.
        * Jaga berat badan ideal agar Indeks Massa Tubuh (BMI) tetap berada dalam rentang normal (18.5 - 24.9).
        * Lakukan aktivitas fisik atau olahraga rutin minimal 150 menit per minggu sesuai anjuran WHO.
        """)
