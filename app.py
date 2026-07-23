import streamlit as st
import numpy as np
import pickle
import pandas as pd

# 1. KONFIGURASI HALAMAN
st.set_page_config(layout="wide", page_title="Prediksi Diabetes - Naive Bayes")

# 2. LOAD MODEL NAIVE BAYES & SCALER STERIL
@st.cache_resource
def load_model():
    with open('diabetes_prediction_NB.pkl', 'rb') as f:
        return pickle.load(f)

model_bundle = load_model()
model = model_bundle["model"]
scaler = model_bundle["scaler"]

st.title("Aplikasi Prediksi Diabetes - Algoritma Naive Bayes")
st.write("Masukkan data medis Anda pada form di sebelah kiri untuk melihat hasil analisis prediksi.")
st.write("---")

# MEMBUAT TAMPILAN 2 KOLOM
col1, col2 = st.columns([4, 5], gap="large")

# ==================== KOLOM 1: FORM INPUT DATA ====================
with col1:
    st.subheader("Form Data Medis Pasien")
    
    with st.form("form_diabetes_kamu"):
        pregnancies = st.number_input('Pregnancies (Jumlah Kehamilan)', min_value=0, max_value=20, step=1, value=0)
        glucose = st.number_input('Glucose (Kadar Glukosa mg/dL)', min_value=40, max_value=500, value=100) 
        blood_pressure = st.number_input('Blood Pressure (Tekanan Darah mmHg)', min_value=40, max_value=240, value=70) 
        skin_thickness = st.number_input('Skin Thickness (Ketebalan Kulit mm)', min_value=10, max_value=100, value=20)
        insulin = st.number_input('Insulin (mIU/L)', min_value=15, max_value=900, value=80) 
        bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=10.0, max_value=70.0, format="%.1f", value=22.5)
        dpf = st.number_input('Diabetes Pedigree Function (Skor Genetik)', min_value=0.001, max_value=3.000, format="%.3f", value=0.350)
        age = st.number_input('Age (Umur Tahun)', min_value=10, max_value=120, step=1, value=25)
        
        submit = st.form_submit_button("Proses Analisis Medis")

# ==================== KOLOM 2: DUA TABEL INDIKATOR ACUAN ====================
with col2:
    st.subheader("Panduan & Acuan Pembanding Medis")
    st.info("Pilih tab di bawah untuk melihat acuan medis internasional (WHO) atau profil statistik dataset pelatihan AI:")

    # Menggunakan fitur Tab agar tampilan rapi
    tab1, tab2 = st.tabs(["📋 Tabel 1: Acuan WHO/ADA", "📊 Tabel 2: Statistik Dataset PIMA"])

    with tab1:
        st.markdown("**Standar Klinis Internasional (WHO / ADA)**")
        st.markdown("""
        | Fitur Medis | Normal / Rendah Risiko | Waspada / Tinggi Risiko |
        | :--- | :--- | :--- |
        | **Pregnancies** | 0 - 3 kali | ≥ 4 kali |
        | **Glucose** | < 140 mg/dL | ≥ 140 mg/dL |
        | **Blood Pressure** | < 80 mmHg | ≥ 80 mmHg |
        | **Skin Thickness** | < 30 mm | ≥ 30 mm |
        | **Insulin** | < 160 mIU/L | ≥ 160 mIU/L |
        | **BMI** | < 25.0 kg/m² | ≥ 25.0 kg/m² |
        | **Diabetes Pedigree** | < 0.500 | ≥ 0.500 |
        | **Age** | < 30 tahun | ≥ 30 tahun |
        
        > *Panduan univariat medis untuk indikator tunggal.*
        """)

    with tab2:
        st.markdown("**Profil Statistik Dataset Penelitian (744 Data PIMA)**")
        st.markdown("""
        | Fitur Medis | Rata-rata (Mean) | Standar Deviasi | Nilai Min - Max |
        | :--- | :--- | :--- | :--- |
        | **Pregnancies** | 3.8 kali | 3.3 | 0 - 17 kali |
        | **Glucose** | 121.7 mg/dL | 30.5 | 44 - 199 mg/dL |
        | **Blood Pressure** | 72.4 mmHg | 12.1 | 24 - 122 mmHg |
        | **Skin Thickness** | 29.2 mm | 8.9 | 7 - 99 mm |
        | **Insulin** | 141.8 mIU/L | 86.2 | 14 - 846 mIU/L |
        | **BMI** | 32.5 kg/m² | 6.9 | 18.2 - 67.1 kg/m² |
        | **Diabetes Pedigree** | 0.472 | 0.33 | 0.078 - 2.420 |
        | **Age** | 33.2 tahun | 11.8 | 21 - 81 tahun |
        
        > *Pola statistik riil yang dipelajari oleh model Naive Bayes secara multivariat.*
        """)

# ==================== PROSES PREDIKSI & OUTPUT HASIL ====================
if submit:
    kolom_asli = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=kolom_asli)
    
    # 1. Feature Scaling steril
    features_scaled = scaler.transform(input_df)
    
    # 2. Prediksi Klasifikasi dan Probabilitas
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    prob_positif = probabilities[1] * 100
    
    st.write("---")
    st.subheader("Hasil Analisis Model Sistem")
    
    st.metric(label="Estimasi Tingkat Probabilitas Risiko Diabetes", value=f"{prob_positif:.1f}%")
    
    if prediction == 1:
        st.error(f"Hasil Analisis Medis: Pasien Terindikasi POSITIF Diabetes Mellitus (Probabilitas: {prob_positif:.1f}%)")
        st.markdown("""
        **Rekomendasi Akademis & Medis:** 
        * Segera lakukan pemeriksaan penunjang lanjutan (seperti tes HbA1c) dan konsultasi dengan dokter spesialis penyakit dalam (Endokrinolog).
        * Mulai batasi asupan karbohidrat sederhana dan makanan dengan indeks glikemik tinggi.
        """)
    else:
        st.success(f"Hasil Analisis Medis: Pasien NEGATIF / Tidak Terindikasi Diabetes Mellitus (Probabilitas Risiko: {prob_positif:.1f}%)")
        st.markdown("""
        **Rekomendasi Akademis & Medis:** 
        * Pertahankan pola hidup sehat yang dijalankan saat ini.
        * Jaga berat badan ideal agar Indeks Massa Tubuh (BMI) tetap berada dalam rentang normal (18.5 - 24.9).
        * Lakukan aktivitas fisik atau olahraga rutin minimal 150 menit per minggu sesuai anjuran WHO.
        """)
