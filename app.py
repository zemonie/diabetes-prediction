import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Konfigurasi halaman diletakkan di paling atas agar tampilan melebar
st.set_page_config(layout="wide")

# 1. LOAD MODEL & SCALER STERIL
with open('diabetes_prediction_KNN.pkl', 'rb') as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
scaler = model_bundle["scaler"]

st.title("Aplikasi Prediksi Diabetes - Algoritma K-Nearest Neighbors")
st.write("Masukkan data medis Anda pada form di sebelah kiri untuk melihat hasil analisis prediksi.")
st.write("---")

# MEMBUAT TAMPILAN 2 KOLOM
col1, col2 = st.columns([4, 5], gap="large")

# ==================== KOLOM 1: FORM INPUT DATA ====================
with col1:
    st.subheader("Form Data Medis")
    
    with st.form("form_diabetes_kamu"):
        # Nilai default (value) diubah kembali menjadi 0 / 0.0 sesuai permintaan
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
# ==================== KOLOM 2: TABEL INDIKATOR REFERENSI ====================
# ==================== KOLOM 2: TABEL INDIKATOR REFERENSI ====================
with col2:
    st.subheader("Panduan Referensi & Pola Model")
    st.info("Aplikasi ini menggabungkan standar medis klinis dengan pola statistik Machine Learning.")
    
    # TAB 1: STANDAR MEDIS (WHO/PERKENI)
    st.markdown("**1. Standar Klinis Medis (WHO/PERKENI)**")
    st.markdown("""
    | Fitur Medis | Rentang Normal / Sehat | Rentang Risiko / Waspada |
    | :--- | :--- | :--- |
    | **Glucose** | < 140 mg/dL | ≥ 140 mg/dL |
    | **Blood Pressure** | < 80 mmHg | ≥ 80 mmHg |
    | **BMI** | < 25.0 | ≥ 25.0 |
    | **Age** | < 30 tahun | ≥ 30 tahun |
    *(Dan fitur medis lainnya)*
    """)
    
    st.divider() # Garis pemisah

    # TAB 2: POLA STATISTIK MODEL (KNN) - INI PENGGANTI TABEL MODEL
    st.markdown("**2. Pola Khas Berdasarkan Data Latih (KNN)**")
    st.caption("Model KNN tidak menggunakan angka batas kaku, melainkan mengenali 'pola gabungan' dari data historis berikut:")
    
    # Membuat 2 kolom kecil untuk membandingkan pola khas
    col_sehat, col_sakit = st.columns(2)
    
    with col_sehat:
        st.success("**🟢 Pola Khas Pasien SEHAT**")
        st.markdown("""
        * **Glucose:** ~110 mg/dL
        * **BMI:** ~23.0
        * **Age:** ~25 tahun
        * **Insulin:** ~80 mIU/L
        * *Kesimpulan: Model mencari kombinasi nilai yang secara bersamaan berada di zona rendah.*
        """)
        
    with col_sakit:
        st.error("**🔴 Pola Khas Pasien DIABETES**")
        st.markdown("""
        * **Glucose:** ~160 mg/dL
        * **BMI:** ~32.0
        * **Age:** ~45 tahun
        * **Insulin:** ~150 mIU/L
        * *Kesimpulan: Model mendeteksi diabetes ketika SEBAGIAN BESAR fitur bergeser ke zona tinggi secara bersamaan.*
        """)

    # KOTAK DISCLAIMER PENTING (WAJIB ADA)
    st.warning("""
    **⚠️ Mengapa Hasil Prediksi Bisa Berbeda dengan Tabel WHO?**
    Tabel WHO bersifat **Univariat** (menilai 1 fitur secara terpisah, misal: Glukosa > 140 = Waspada). 
    Sebaliknya, Model KNN bersifat **Multivariat**. 
    
    *Contoh:* Jika Glukosa Anda 172 (Waspada menurut WHO), tetapi BMI, Usia, dan Tekanan Darah Anda sangat normal, Model KNN akan memprediksi **SEHAT** karena secara statistik, profil gabungan Anda lebih mirip dengan "Pola Khas Pasien Sehat" di atas.
    """)

# ==================== PROSES PREDIKSI & OUTPUT HASIL ====================
if submit:
    kolom_asli = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=kolom_asli)
    
    # ==============================================================================
    # PERBAIKAN KRITIS: Imputasi nilai 0 dengan Median (Sama persis seperti di Colab)
    # ==============================================================================
    cols_medis = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Nilai median ini adalah standar dataset PIMA Indians Diabetes. 
    median_values = {
        'Glucose': 117.0,
        'BloodPressure': 72.0,
        'SkinThickness': 29.0,
        'Insulin': 126.0,
        'BMI': 32.4  
    }
    
    for col in cols_medis:
        # Ganti nilai 0 yang diinput user dengan nilai median yang sesuai
        input_df[col] = input_df[col].replace(0, median_values[col])
    # ==============================================================================
    
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
