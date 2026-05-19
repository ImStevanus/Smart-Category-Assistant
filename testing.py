import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Academic Performance Analyzer | Stevanus",
    page_icon="🎓",
    layout="wide"
)

# 2. CUSTOM CSS (SUPPORT DARK & LIGHT MODE)
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1); 
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px;
        border-radius: 12px;
        color: inherit;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .stSlider, .stNumberInput {
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. FUNGSI LOAD ASSETS
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# 4. KAMUS INFORMASI KLASTER STATIS (SINKRONISASI WARNA DAN TEKS)
def get_cluster_static_info(category_name):
    if category_name == "Tinggi":
        return {
            "label": "🚀 High Achiever (Performa Tinggi)",
            "desc": "Mahasiswa menunjukkan kombinasi performa nilai akademis yang sangat memuaskan di atas rata-rata kelas serta tingkat kehadiran yang sangat konsisten.",
            "saran": "Pertahankan ritme belajar saat ini. Sangat direkomendasikan untuk menjadi asisten dosen atau mentor sebaya.",
            "color": "#2ecc71" # Hijau
        }
    elif category_name == "Menengah":
        return {
            "label": "📊 Steady / Average (Performa Menengah)",
            "desc": "Mahasiswa menunjukkan performa yang cukup stabil di tingkat rata-rata kelompok, namun masih memiliki ruang evaluasi pada nilai ujian utama.",
            "saran": "Fokus meningkatkan pemahaman pada materi ujian tengah semester dan final untuk mendongkrak pencapaian nilai.",
            "color": "#f1c40f" # Kuning
        }
    else: # Berisiko
        return {
            "label": "⚠️ Underperformer (Performa Berisiko)",
            "desc": "Akumulasi perolehan nilai berada di tingkat kritis atau di bawah standar minimal kelulusan. Risiko tinggi mengalami kendala studi.",
            "saran": "Segera jadwalkan sesi bimbingan konseling akademik intensif bersama dosen wali untuk pemulihan nilai.",
            "color": "#e74c3c" # Merah
        }

# 5. SIDEBAR NAVIGATION
with st.sidebar:
    st.title("🎓 Smart Campus AI")
    st.markdown(f"User: **Stevanus**\n\nTema: **Academic Clustering**")
    st.divider()
    menu = st.radio("Navigasi Utama", ["🏠 Dashboard Analisis", "🔍 Prediksi Individu"])
    st.divider()
    uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")

# 6. LOGIKA UTAMA APLIKASI
if model is None or scaler is None:
    st.error("❌ File model (.pkl) tidak ditemukan. Pastikan sudah menjalankan training di Colab!")
else:
    features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']

    # --- MENU 1: DASHBOARD ANALISIS MASSAL ---
    if menu == "🏠 Dashboard Analisis":
        st.title("📊 Dashboard Analisis Performa Mahasiswa")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Preprocessing & Clustering data massal menggunakan K-Means (.pkl)
            df_clean = df[features].fillna(df[features].mean())
            scaled_data = scaler.transform(df_clean)
            df['Cluster'] = model.predict(scaled_data)
            
            # Meranking performa cluster berdasarkan rata-rata nilai final asli dari CSV
            cluster_performance = df.groupby('Cluster')['final_marks'].mean().sort_values(ascending=False)
            cluster_ranks = cluster_performance.index.tolist()
            
            csv_mapping = {}
            if len(cluster_ranks) >= 3:
                csv_mapping[cluster_ranks[0]] = "Tinggi"
                csv_mapping[cluster_ranks[1]] = "Menengah"
                csv_mapping[cluster_ranks[2]] = "Berisiko"
            else:
                for idx, c_id in enumerate(cluster_ranks):
                    csv_mapping[c_id] = "Tinggi" if idx == 0 else "Menengah"

            # Terapkan pemetaan label ke data tabular dashboard
            df['Kategori_Evaluasi'] = df['Cluster'].map(csv_mapping)
            df['Cluster_Name'] = df['Kategori_Evaluasi'].apply(lambda x: get_cluster_static_info(x)['label'])
            
            # PANEL RINGKASAN METRIK
            st.subheader("📌 Ringkasan Data Saat Ini")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Mahasiswa", len(df))
            m2.metric("Rata-rata Final", f"{df['final_marks'].mean():.1f}")
            m3.metric("Rata-rata IPK", f"{df['previous_gpa'].mean():.2f}")
            
            st_categories = df['Kategori_Evaluasi'].nunique()
            m4.metric("Kategori Terdeteksi", f"{st_categories} Kelompok")
            
            st.divider()

            # PANEL VISUALISASI DATA GRAPH
            col_left, col_right = st.columns([6, 4])
            
            with col_left:
                st.subheader("📍 Sebaran Cluster (Midterm vs Final)")
                fig = px.scatter(df, x="midterm_marks", y="final_marks", color="Cluster_Name", 
                                 template="none", color_discrete_map={
                                     get_cluster_static_info("Tinggi")['label']: "#2ecc71",
                                     get_cluster_static_info("Menengah")['label']: "#f1c40f",
                                     get_cluster_static_info("Berisiko")['label']: "#e74c3c"
                                 })
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.subheader("🥧 Proporsi Kelompok")
                fig_pie = px.pie(df, names="Cluster_Name", hole=0.4, template="none",
                                 color="Cluster_Name", color_discrete_map={
                                     get_cluster_static_info("Tinggi")['label']: "#2ecc71",
                                     get_cluster_static_info("Menengah")['label']: "#f1c40f",
                                     get_cluster_static_info("Berisiko")['label']: "#e74c3c"
                                 })
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("📋 Eksplorasi Data Lengkap")
            cl_filter = st.multiselect("Filter Tampilan Kategori:", options=df['Cluster_Name'].unique(), default=df['Cluster_Name'].unique())
            st.dataframe(df[df['Cluster_Name'].isin(cl_filter)], use_container_width=True)
            
        else:
            st.info("Silakan unggah dataset 'student_dropout_behavior_dataset.csv' pada sidebar untuk melihat analisis kelompok.")

    # --- MENU 2: PREDIKSI INDIVIDU BARU (ANTI-BIAS / AMAN NYATA) ---
    else:
        st.title("🔍 Prediksi Kategori Mahasiswa")
        st.write("Masukkan parameter akademis mahasiswa di bawah ini untuk melihat hasil prediksi kelompok:")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                q1 = st.slider("Quiz 1 (0-10)", 0.0, 10.0, 0.0)
                q2 = st.slider("Quiz 2 (0-10)", 0.0, 10.0, 0.0)
                q3 = st.slider("Quiz 3 (0-10)", 0.0, 10.0, 0.0)
                mid = st.number_input("Midterm Marks (0-30)", 0, 30, 0)
            with c2:
                fin = st.number_input("Final Marks (0-50)", 0, 50, 0)
                gpa = st.slider("Previous GPA (0.0-4.0)", 0.0, 4.0, 0.0)
                lec = st.number_input("Lectures Attended (0-12)", 0, 12, 0)
                lab = st.number_input("Labs Attended (0-6)", 0, 6, 0)
            
            submit = st.form_submit_button("🚀 Analisis Performa")

        if submit:
            # HITUNG TOTAL SKOR AKADEMIK SECARA RIIL (Maksimum Teoretis Dataset: 110)
            total_skor_ujian = fin + mid + (q1 + q2 + q3)
            
            # Sistem Pengondisian Batas Nilai Mutlak yang Sesuai Aturan Akademik Kelas
            if (fin == 0 and mid == 0) or total_skor_ujian <= 45.0 or lec <= 4:
                final_category = "Berisiko"
            elif total_skor_ujian <= 76.0:
                final_category = "Menengah"
            else:
                final_category = "Tinggi"
            
            # Panggil informasi berdasarkan keputusan akhir klasifikasi yang adil
            info = get_cluster_static_info(final_category)
            
            # CETAK KARTU HASIL PREDIKSI BARU
            st.markdown(f"""
                <div style="background-color:{info['color']}; padding:30px; border-radius:15px; text-align:center; color:white;">
                    <h1 style="color:white; margin:0;">{info['label']}</h1>
                    <p style="font-size:1.2em; margin:15px 0;">{info['desc']}</p>
                    <div style="background-color:rgba(0,0,0,0.15); padding:15px; border-radius:10px;">
                        <b>💡 Rekomendasi UAS:</b> {info['saran']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# 7. FOOTER APLIKASI
st.divider()
st.caption(f"UAS Pemrograman AI - Stevanus - K-Means Clustering Teroptimasi")
