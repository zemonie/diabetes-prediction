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

st.title("Aplikasi Prediksi Diabetes - Algoritma Support Vector Machine")
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
with col2:
    st.subheader("Panduan Indikator Medis Faktual")
    st.info("Gunakan tabel referensi di bawah ini sebagai panduan standar internasional (WHO/ADA) saat mengisi form:")
    
    # REVISI: Menyesuaikan ambang batas agar lebih akurat dengan karakteristik dataset dan logika model multivariat
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
        'BMI': 32.0
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
