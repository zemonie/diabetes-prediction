import streamlit as st
import numpy as np
import pickle
import pandas as pd

# 1. LOAD MODEL & SCALER STERIL
with open('diabetes_prediction.pkl', 'rb') as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
scaler = model_bundle["scaler"]

# PERBAIKAN CELAH 1: Judul disesuaikan dengan Algoritma Terbaik di Sel 13 Colab
st.title("Aplikasi Prediksi Diabetes - Algoritma Random Forest")
st.write("Masukkan data medis Anda di bawah ini untuk melihat hasil analisis prediksi.")

# TABEL INDIKATOR REFERENSI FAKTUAL
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
            "0 - 3 kali", "< 140 mg/dL", "< 80 mmHg", "10 - 29 mm", "< 160 mIU/L", "18.5 - 24.9", "< 0.500", "21 - 30 tahun"
        ],
        "Kriteria Risiko / Diabetes": [
            " > 4 kali", ">= 200 mg/dL (Diabetes) | 140-199 (Pre)", ">= 80 mmHg", ">= 33 mm", "> 160 mIU/L", ">= 30.0 (Obesitas)", ">= 0.500", "> 30 tahun"
        ]
    }
    st.table(pd.DataFrame(data_indikator))

# FORM INPUT DATA MEDIS
with st.form("form_diabetes_kamu"):
    pregnancies = st.number_input('Pregnancies (Jumlah Kehamilan)', min_value=0, max_value=20, step=1, value=0)
    glucose = st.number_input('Glucose (Kadar Glukosa)', min_value=0, max_value=200, value=0)
    blood_pressure = st.number_input('Blood Pressure (Tekanan Darah)', min_value=0, max_value=150, value=0)
    skin_thickness = st.number_input('Skin Thickness (Ketebalan Kulit)', min_value=0, max_value=100, value=0)
    insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0)
    bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=0.0, max_value=70.0, format="%.1f", value=0.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, format="%.3f", value=0.085)
    age = st.number_input('Age (Umur)', min_value=1, max_value=120, step=1, value=21)
    
    submit = st.form_submit_button("Proses Analisis Medis")

if submit:
    # PERBAIKAN CELAH 3: Membungkus input langsung ke DataFrame dengan susunan kolom asli (Steril)
    kolom_asli = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=kolom_asli)
    
    # PERBAIKAN CELAH 2 (CRITICAL): Imputasi nilai 0 otomatis di latar belakang menggunakan MEDIAN riil dataset Pima
    # Nilai di bawah ini diambil dari nilai tengah resmi dataset pasca cleaning Sel 3
    median_dataset = {
        'Glucose': 117.0,
        'BloodPressure': 72.0,
        'SkinThickness': 23.0,
        'Insulin': 30.5,
        'BMI': 32.0
    }
    
    for kolom, nilai_median in median_dataset.items():
        if input_df.loc[0, kolom] == 0:
            input_df.loc[0, kolom] = nilai_median

    # Jalankan proses Scaling secara terstruktur
    features_scaled = scaler.transform(input_df)
    
    # Jalankan Prediksi Klasifikasi
    prediction = model.predict(features_scaled)[0]
    
    # TAMPILAN OUTPUT PREDIKSI DIJAMIN AKURAT
    st.write("---")
    if prediction == 1:
        st.error("Hasil Analisis Medis: Pasien Terindikasi POSITIF Diabetes Mellitus")
        st.markdown("**Rekomendasi:** Segera lakukan konsultasi lebih lanjut dengan dokter spesialis penyakit dalam (Endokrinolog) dan kurangi konsumsi glukosa tinggi.")
    else:
        st.success("Hasil Analisis Medis: Pasien NEGATIF / Tidak Terindikasi Diabetes Mellitus")
        st.markdown("**Rekomendasi:** Pertahankan pola hidup sehat, jaga indeks massa tubuh (BMI) ideal, dan lakukan olahraga rutin.")
