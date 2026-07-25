import streamlit as st
import numpy as np
import pickle
import pandas as pd

# 1. KONFIGURASI HALAMAN
st.set_page_config(layout="wide", page_title="Prediksi Diabetes - Naive Bayes")

# STYLING GLOBAL CSS FIX - CLEAN CARD DESIGN
st.markdown("""
    <style>
    /* 1. Atur padding utama halaman */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 95% !important;
    }
    
    /* 2. Rapatkan jarak antar widget */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.2rem !important;
    }
    
    /* 3. Atur form input agar ringkas */
    div[data-testid="stForm"] {
        padding: 10px !important;
        border-radius: 8px !important;
    }
    .stNumberInput {
        margin-bottom: -6px !important;
    }
    
    /* 4. Styling Card Output Prediksi agar Rapi & Elegan */
    .result-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 15px;
        margin-top: 6px;
    }
    .result-header {
        font-size: 15px;
        font-weight: bold;
        color: #212529;
        margin-bottom: 4px;
    }
    .rec-title {
        font-size: 12px;
        font-weight: bold;
        color: #333;
        margin-top: 6px;
        margin-bottom: 2px;
    }
    .rec-list {
        font-size: 11px;
        color: #495057;
        margin: 0;
        padding-left: 15px;
    }
    
    /* 5. Formatting Tabel di Kolom Kanan */
    table {
        font-size: 13px !important;
        margin-bottom: 2px !important;
        width: 100% !important;
    }
    th, td {
        padding: 3px 6px !important;
    }
    .table-title {
        font-size: 14px !important;
        font-weight: bold;
        margin-top: 3px;
        margin-bottom: 2px;
    }
    .table-note {
        font-size: 11px !important;
        color: #666;
        margin-top: 1px;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. LOAD MODEL NAIVE BAYES & SCALER STERIL
@st.cache_resource
def load_model():
    with open('diabetes_prediction_NB.pkl', 'rb') as f:
        return pickle.load(f)

model_bundle = load_model()
model = model_bundle["model"]
scaler = model_bundle["scaler"]

# HEADER UTAMA
st.title("Aplikasi Prediksi Diabetes - Algoritma Naive Bayes")
st.caption("Masukkan data medis Anda pada form di sebelah kiri untuk melihat hasil analisis prediksi.")

# MEMBUAT TAMPILAN 2 KOLOM
col1, col2 = st.columns([5, 5], gap="medium")

# ==================== KOLOM 1: FORM INPUT + OUTPUT PREDIKSI ====================
with col1:
    st.markdown("<h4 style='margin-bottom: 4px; margin-top: 0px;'>Form Data Medis Pasien</h4>", unsafe_allow_html=True)
    
    with st.form("form_diabetes_kamu"):
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            pregnancies = st.number_input('Pregnancies (Kehamilan)', min_value=0, max_value=20, step=1, value=0)
            glucose = st.number_input('Glucose (Glukosa mg/dL)', min_value=40, max_value=500, value=100) 
            blood_pressure = st.number_input('Blood Pressure (Tekanan Darah)', min_value=40, max_value=240, value=70) 
            skin_thickness = st.number_input('Skin Thickness (Ketebalan Kulit)', min_value=10, max_value=100, value=20)
            
        with f_col2:
            insulin = st.number_input('Insulin (mIU/L)', min_value=15, max_value=900, value=80) 
            bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=10.0, max_value=70.0, format="%.1f", value=22.5)
            dpf = st.number_input('Diabetes Pedigree (Skor Genetik)', min_value=0.001, max_value=3.000, format="%.3f", value=0.350)
            age = st.number_input('Age (Umur Tahun)', min_value=10, max_value=120, step=1, value=25)
        
        submit = st.form_submit_button("Proses Analisis Medis", use_container_width=True)

    # OUTPUT HASIL PREDIKSI (DESAIN CARD CLEAN & RAPI)
    if submit:
        kolom_asli = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=kolom_asli)
        
        # 1. Feature Scaling steril
        features_scaled = scaler.transform(input_df)
        
        # 2. Prediksi Klasifikasi dan Probabilitas
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        prob_positif = probabilities[1] * 100
        
        # Bungkusan Card Hasil
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Header + Probabilitas Sejajar
        res_col1, res_col2 = st.columns([3, 2])
        with res_col1:
            st.markdown('<div class="result-header">Hasil Analisis Model Sistem</div>', unsafe_allow_html=True)
        with res_col2:
            st.metric(label="Probabilitas Risiko", value=f"{prob_positif:.1f}%")
        
        # Status Diagnosa
        if prediction == 1:
            st.error(f"**POSITIF** — Pasien Terindikasi Diabetes Mellitus ({prob_positif:.1f}%)")
            st.markdown('<div class="rec-title">Rekomendasi Medis:</div>', unsafe_allow_html=True)
            st.markdown("""
            <ul class="rec-list">
                <li>Segera konsultasi ke dokter spesialis penyakit dalam (Endokrinolog).</li>
                <li>Batasi asupan karbohidrat sederhana dan makanan berglukosa tinggi.</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.success(f"**NEGATIF** — Pasien Tidak Terindikasi Diabetes Mellitus ({prob_positif:.1f}%)")
            st.markdown('<div class="rec-title">Rekomendasi Medis:</div>', unsafe_allow_html=True)
            st.markdown("""
            <ul class="rec-list">
                <li>Pertahankan pola hidup sehat dan pertahankan BMI ideal (18.5 - 24.9).</li>
                <li>Lakukan aktivitas fisik/olahraga rutin minimal 150 menit per minggu.</li>
            </ul>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== KOLOM 2: TABEL ACUAN PEMBANDING ====================
with col2:
    st.markdown("<h4 style='margin-bottom: 4px; margin-top: 0px;'>Panduan & Acuan Pembanding Medis</h4>", unsafe_allow_html=True)
    st.info("Berikut acuan medis internasional & ambang batas keputusan model AI:")

    # TABEL 1: ACUAN WHO
    st.markdown('<p class="table-title">📋 TABEL 1: ACUAN WHO / ADA</p>', unsafe_allow_html=True)
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
    """)
    st.markdown('<p class="table-note">*Catatan: Standar acuan medis klinis univariat.</p>', unsafe_allow_html=True)

    # TABEL 2: AMBANG BATAS AI
    st.markdown('<p class="table-title">📊 TABEL 2: AMBANG BATAS MODEL AI KITA</p>', unsafe_allow_html=True)
    st.markdown("""
    | Fitur Medis | Normal / Rendah Risiko | Waspada / Tinggi Risiko |
    | :--- | :--- | :--- |
    | **Pregnancies** | ≤ 5 kali | > 5 kali |
    | **Glucose** | ≤ 160 mg/dL | > 160 mg/dL |
    | **Blood Pressure** | ≤ 80 mmHg | > 80 mmHg |
    | **Skin Thickness** | ≤ 30 mm | > 30 mm |
    | **Insulin** | ≤ 160 mIU/L | > 160 mIU/L |
    | **BMI** | ≤ 25.5 kg/m² | > 25.5 kg/m² |
    | **Diabetes Pedigree** | ≤ 0.500 | > 0.500 |
    | **Age** | ≤ 30 tahun | > 30 tahun |
    """)
    st.markdown('<p class="table-note">*Catatan: Batas toleransi multivariat tertinggi (Probabilitas ≤ 49.7%).</p>', unsafe_allow_html=True)
