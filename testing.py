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

# 4. FUNSI ADAPTIF: Pemetaan Kategori Sesuai Karakteristik Dataset (Skala Nilai Ujian Kumulatif Max: 110)
def get_accurate_cluster_mapping(model, scaler, features):
    if model is None or scaler is None:
        return {0: "Tinggi", 1: "Menengah", 2: "Berisiko"}
    
    # Ambil titik pusat klaster (Cluster Centers) dan balikkan ke skala asli
    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    df_centroids = pd.DataFrame(centroids_original, columns=features)
    
    # Hitung performa akademis kumulatif berdasarkan komponen ujian utama
    df_centroids['performance_score'] = (
        df_centroids['final_marks'] + 
        df_centroids['midterm_marks'] + 
        (df_centroids['quiz1_marks'] + df_centroids['quiz2_marks'] + df_centroids['quiz3_marks'])
    )
    
    mapping = {}
    for cluster_idx, row in df_centroids.iterrows():
        score = row['performance_score']
        
        # Penyesuaian Kategori Berdasarkan Distribusi Skor Total Dataset
        if score < 55.0:       # Mengumpulkan nilai < 50% dari total poin ujian
            mapping[cluster_idx] = "Berisiko"
        elif score < 78.0:     # Berada di rentang nilai rata-rata kelompok (50% - 70%)
            mapping[cluster_idx] = "Menengah"
        else:                  # Mengumpulkan nilai > 70% dari total poin ujian
            mapping[cluster_idx] = "Tinggi"
            
    return mapping

def get_cluster_info(cluster_num, mapping_dict):
    status = mapping_dict.get(cluster_num, "Menengah")
    
    if status == "Tinggi":
        return {
            "label": "🚀 High Achiever (Performa Tinggi)",
            "desc": "Mahasiswa dengan akumulasi nilai ujian yang sangat baik (di atas 70% dari total bobot nilai) dan konsisten.",
            "saran": "Pertahankan ritme belajar. Sangat direkomendasikan menjadi asisten dosen atau mentor sebaya.",
            "color": "#2ecc71"
        }
    elif status == "Menengah":
        return {
            "label": "📊 Steady / Average (Performa Menengah)",
            "desc": "Mahasiswa menunjukkan performa stabil pada tingkat rata-rata kelompok, namun memiliki ruang peningkatan pada aspek nilai ujian utama.",
            "saran": "Fokus meningkatkan nilai ujian utama dan tugas harian untuk mengamankan nilai akhir.",
            "color": "#f1c40f"
        }
    else: # Berisiko
        return {
            "label": "⚠️ Underperformer (Performa Berisiko)",
            "desc": "Akumulasi nilai ujian sangat rendah (di bawah 50% dari total bobot nilai). Berisiko tinggi mengalami kendala kelulusan.",
            "saran": "Segera jadwalkan sesi bimbingan konseling akademik untuk pemulihan nilai.",
            "color": "#e74c3c"
        }

# 5. SIDEBAR NAVIGATION & FILE UPLOADER
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
    # 8 Fitur utama yang digunakan sesuai dengan model training
    features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']
    
    # Jalankan pemetaan otomatis berdasarkan matematika model pkl riil
    cluster_mapping = get_accurate_cluster_mapping(model, scaler, features)

    # --- MENU 1: DASHBOARD ANALISIS ---
    if menu == "🏠 Dashboard Analisis":
        st.title("📊 Dashboard Analisis Performa Mahasiswa")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Preprocessing & Clustering data baru
            df_clean = df[features].fillna(df[features].mean())
            scaled_data = scaler.transform(df_clean)
            df['Cluster'] = model.predict(scaled_data)
            
            # PANEL RINGKASAN METRIK
            st.subheader("📌 Ringkasan Data Saat Ini")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Mahasiswa", len(df))
            m2.metric("Rata-rata Final", f"{df['final_marks'].mean():.1f}")
            m3.metric("Rata-rata IPK", f"{df['previous_gpa'].mean():.2f}")
            m4.metric("Kategori Cluster Terdeteksi", len(df['Cluster'].unique()))
            
            st.divider()

            # PANEL VISUALISASI DATA
            col_left, col_right = st.columns([6, 4])
            
            with col_left:
                st.subheader("📍 Sebaran Cluster (Midterm vs Final)")
                fig = px.scatter(df, x="midterm_marks", y="final_marks", color="Cluster", 
                                 template="none", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.subheader("🥧 Proporsi Kelompok")
                df['Cluster_Name'] = df['Cluster'].apply(lambda x: get_cluster_info(x, cluster_mapping)['label'])
                fig_pie = px.pie(df, names="Cluster_Name", hole=0.4, template="none")
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("📋 Eksplorasi Data Lengkap")
            cl_filter = st.multiselect("Filter Tampilan Cluster:", options=sorted(df['Cluster'].unique()), default=df['Cluster'].unique())
            st.dataframe(df[df['Cluster'].isin(cl_filter)], use_container_width=True)
            
        else:
            st.info("Silakan unggah dataset 'student_dropout_behavior_dataset.csv' pada sidebar untuk melihat analisis kelompok.")

    # --- MENU 2: PREDIKSI INDIVIDU ---
    else:
        st.title("🔍 Prediksi Kategori Mahasiswa")
        st.write("Masukkan parameter akademis mahasiswa di bawah ini untuk melihat hasil prediksi kelompok:")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                q1 = st.slider("Quiz 1", 0.0, 10.0, 0.0)
                q2 = st.slider("Quiz 2", 0.0, 10.0, 0.0)
                q3 = st.slider("Quiz 3", 0.0, 10.0, 0.0)
                mid = st.number_input("Midterm Marks (0-30)", 0, 30, 0)
            with c2:
                fin = st.number_input("Final Marks (0-50)", 0, 50, 0)
                gpa = st.slider("Previous GPA (0.0-4.0)", 0.0, 4.0, 0.0)
                lec = st.number_input("Lectures Attended (0-12)", 0, 12, 0)
                lab = st.number_input("Labs Attended (0-6)", 0, 6, 0)
            
            submit = st.form_submit_button("🚀 Analisis Performa")

        if submit:
            # Mengemas input ke dalam bentuk dataframe sesuai urutan fitur training
            input_df = pd.DataFrame([[q1, q2, q3, mid, fin, gpa, lec, lab]], columns=features)
            scaled_input = scaler.transform(input_df)
            res = model.predict(scaled_input)[0]
            
            # Panggil info klasifikasi berdasarkan perhitungan dinamis dari model pkl
            info = get_cluster_info(res, cluster_mapping)
            
            # Tampilan hasil prediksi berupa kartu berwarna yang informatif
            st.markdown(f"""
                <div style="background-color:{info['color']}; padding:30px; border-radius:15px; text-align:center; color:white;">
                    <h1 style="color:white; margin:0;">{info['label']}</h1>
                    <p style="font-size:1.2em; margin:15px 0;">{info['desc']}</p>
                    <div style="background-color:rgba(0,0,0,0.15); padding:15px; border-radius:10px;">
                        <b>💡 Rekomendasi:</b> {info['saran']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# 7. FOOTER APLIKASI
st.divider()
st.caption(f"UAS Pemrograman AI - Stevanus - {len(features)} Fitur Teranalisis Berbasis Dataset Riil")
