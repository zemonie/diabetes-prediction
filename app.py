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
    st.subheader("📝 Form Data Medis")
    
    with st.form("form_diabetes_kamu"):
        # Batasan min_value di sini sudah melindungi sistem dari angka 0 yang tidak logis secara klinis
        pregnancies = st.number_input('Pregnancies (Jumlah Kehamilan)', min_value=0, max_value=20, step=1, value=0)
        glucose = st.number_input('Glucose (Kadar Glukosa)', min_value=30, max_value=500, value=100) 
        blood_pressure = st.number_input('Blood Pressure (Tekanan Darah)', min_value=40, max_value=240, value=80) 
        skin_thickness = st.number_input('Skin Thickness (Ketebalan Kulit)', min_value=5, max_value=100, value=20)
        insulin = st.number_input('Insulin', min_value=5, max_value=900, value=80) 
        bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=10.0, max_value=70.0, format="%.1f", value=22.5)
        dpf = st.number_input('Diabetes Pedigree Function (Skor Genetik)', min_value=0.078, max_value=3.000, format="%.3f", value=0.250)
        age = st.number_input('Age (Umur)', min_value=21, max_value=120, step=1, value=25)
        
        submit = st.form_submit_button("Proses Analisis Medis")

# ==================== KOLOM 2: TABEL INDIKATOR REFERENSI ====================
with col2:
    st.subheader("📊 Panduan Indikator Medis Faktual")
    st.info("Gunakan tabel referensi di bawah ini sebagai panduan standar internasional (WHO/ADA) saat mengisi form:")
    
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
            "0 - 3 kali", "< 140 mg/dL", "< 80 mmHg", "10 - 29 mm", "< 160 mIU/L", "18.5 - 24.9", "< 0.500", "21 - 30 tahun"
        ],
        "Kriteria Risiko / Diabetes": [
            "> 4 kali", ">= 200 mg/dL (Diabetes) | 140-199 (Pre)", ">= 80 mmHg", ">= 33 mm", "> 160 mIU/L", ">= 30.0 (Obesitas)", ">= 0.500", "> 30 tahun"
        ]
    }
    st.table(pd.DataFrame(data_indikator))

# ==================== PROSES PREDIKSI & OUTPUT HASIL ====================
if submit:
    kolom_asli = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=kolom_asli)
    
    # Fitur Scaling secara steril menggunakan objek scaler bawaan dari pkl
    features_scaled = scaler.transform(input_df)
    
    # Prediksi Klasifikasi
    prediction = model.predict(features_scaled)[0]
    
    st.write("### 📢 Hasil Analisis Model Sistem")
    
    if prediction == 1:
        st.error("🚨 **Hasil Analisis Medis:** Pasien Terindikasi **POSITIF** Diabetes Mellitus")
        st.markdown("""
        **Rekomendasi Akademis & Medis:** 
        * Segera lakukan pemeriksaan penunjang lanjutan (seperti tes HbA1c) dan konsultasi dengan dokter spesialis penyakit dalam (Endokrinolog).
        * Mulai batasi asupan karbohidrat sederhana dan glukosa tinggi.
        """)
    else:
        st.success("✅ **Hasil Analisis Medis:** Pasien **NEGATIF** / Tidak Terindikasi Diabetes Mellitus")
        st.markdown("""
        **Rekomendasi Akademis & Medis:** 
        * Pertahankan pola hidup sehat saat ini.
        * Jaga berat badan ideal agar Indeks Massa Tubuh (BMI) tetap berada dalam rentang normal.
        * Lakukan aktivitas fisik atau olahraga rutin minimal 150 menit per minggu.
        """)
