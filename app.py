# ==================== KOLOM 2: DUA TABEL INDIKATOR ACUAN ====================
with col2:
    st.subheader("Panduan & Acuan Pembanding Medis")
    st.info("Klik salah satu tombol besar di bawah ini untuk menampilkan tabel acuan:")

    # Inisialisasi state untuk menyimpan pilihan tabel (Default: Tabel 1)
    if "pilihan_tabel" not in st.session_state:
        st.session_state.pilihan_tabel = "WHO"

    # Styling CSS untuk membuat tombol st.button berukuran BESAR & Mencolok
    st.markdown("""
        <style>
        /* Mengubah gaya tombol Streamlit agar berukuran besar dan jelas */
        div.stButton > button {
            width: 100%;
            height: 60px;
            font-size: 18px !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            border: 2px solid #2e7d32 !important;
            transition: all 0.3s ease;
        }
        </style>
    """, unsafe_allow_html=True)

    # Membuat 2 Kolom Tombol Berdampingan
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("📋 TABEL 1: ACUAN WHO / ADA", use_container_width=True, type="primary" if st.session_state.pilihan_tabel == "WHO" else "secondary"):
            st.session_state.pilihan_tabel = "WHO"

    with btn_col2:
        if st.button("📊 TABEL 2: AMBANG BATAS MODEL AI KITA", use_container_width=True, type="primary" if st.session_state.pilihan_tabel == "AI" else "secondary"):
            st.session_state.pilihan_tabel = "AI"

    st.write("") # Jarak pemisah kecil

    # MENAMPILKAN TABEL SESUAI TOMBOL YANG DIKLIK
    if st.session_state.pilihan_tabel == "WHO":
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
        
        > *Catatan: Standar acuan medis klinis univariat.*
        """)

    elif st.session_state.pilihan_tabel == "AI":
        st.markdown("**Batas Toleransi Keputusan Model Naive Bayes (Sistem)**")
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
        
        > *Catatan: Nilai pada kolom 'Normal / Rendah Risiko' di atas merupakan batas toleransi multivariat tertinggi (Probabilitas ≤ 49.7%). Jika kombinasi fitur melebihi angka tersebut, model akan mengklasifikasikannya sebagai POSITIF.*
        """)
