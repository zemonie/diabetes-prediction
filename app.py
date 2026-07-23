import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Konfigurasi halaman diletakkan di paling atas agar tampilan melebar
st.set_page_config(layout="wide", page_title="Prediksi Diabetes - KNN")

# 1. LOAD MODEL & SCALER STERIL
@st.cache_resource
def load_model():
    with open('diabetes_prediction_KNN.pkl', 'rb') as f:
        return pickle.load(f)

model_bundle = load_model()
model = model_bundle["model"]
scaler = model_bundle["scaler"]

st.title("Aplikasi Prediksi Diabetes - Algoritma K-Nearest Neighbors")
st.write("Masukkan data medis Anda pada form di sebelah kiri untuk melihat hasil analisis prediksi.")
st.write("---")

# MEMBUAT TAMPILAN 2 KOLOM
col1, col2 = st.columns([4, 5], gap="large")

# ==================== KOLOM 1: FORM INPUT DATA ====================
with col1:
    st.subheader("Form Data Medis Pasien")
    
    with st.form("form_diabetes_kamu"):
        # Default value diatur ke nilai rata-rata/normal manusia realistis (Bukan 0)
        pregnancies = st.number_input('Pregnancies (Jumlah Kehamilan)', min_value=0, max_value=20, step=1, value=0)
        glucose = st.number_input('Glucose (Kadar Glukosa mg/dL)', min_value=40, max_value=500, value=100) 
        blood_pressure = st.number_input('Blood Pressure (Tekanan Darah mmHg)', min_value=40, max_value=240, value=70) 
        skin_thickness = st.number_input('Skin Thickness (Ketebalan Kulit mm)', min_value=10, max_value=100, value=20)
        insulin = st.number_input('Insulin (mIU/L)', min_value=15, max_value=900, value=80) 
        bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=10.0, max_value=70.0, format="%.1f", value=22.5)
        dpf = st.number_input('Diabetes Pedigree Function (Skor Genetik)', min_value=0.001, max_value=3.000, format="%.3f", value=0.350)
        age = st.number_input('Age (Umur Tahun)', min_value=10, max_value=120, step=1, value=25)
        
        submit = st.form_submit_button("Proses Analisis Medis")

# ==================== KOLOM 2: TABEL INDIKATOR REFERENSI ====================
with col2:
    st.subheader("Panduan Indikator Medis Faktual")
    st.info("Gunakan tabel referensi di bawah ini sebagai panduan standar internasional (WHO/ADA) saat mengisi form:")
    
    st.markdown("""
    | Fitur Medis | Rentang Rendah Risiko | Rentang Perlu Diwaspadai (Risiko) |
    | :--- | :--- | :--- |
    | **Pregnancies** (Jumlah Kehamilan) | 0 - 3 kali | ≥ 4 kali |
    | **Glucose** (Kadar Glukosa) | < 140 mg/dL | ≥ 140 mg/dL (Waspada) / ≥ 200 mg/dL (Risiko Tinggi) |
    | **Blood Pressure** (Tekanan Darah) | < 80 mmHg | ≥ 80 mmHg |
    | **Skin Thickness** (Ketebalan Kulit) | < 30 mm | ≥ 30 mm |
    | **Insulin** (Kadar Insulin) | < 160 mIU/L | ≥ 160 mIU/L |
    | **BMI** (Indeks Massa Tubuh) | < 25.0 | ≥ 25.0 (Overweight/Obesitas) |
    | **Diabetes Pedigree Function** | < 0.500 | ≥ 0.500 |
    | **Age** (Umur) | < 30 tahun | ≥ 30 tahun |
    
    > *Catatan: Model AI menganalisis kombinasi dari semua fitur di atas secara bersamaan (multivariat), bukan hanya satu fitur tunggal. Nilai di ambang batas (misal: Glukosa 170-an) yang disertai faktor risiko lain akan secara signifikan meningkatkan skor prediksi.*
    """)

# ==================== PROSES PREDIKSI & OUTPUT HASIL ====================
if submit:
    kolom_asli = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=kolom_asli)
    
    # Feature Scaling steril menggunakan objek scaler dari pickle
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
