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

# 4. FUNGSI PEMETAAN OTOMATIS BERDASARKAN MODE PARAMETER YANG DIPILIH
def get_model_cluster_mapping(model, scaler, features, mode):
    if model is None or scaler is None:
        return {0: "Tinggi", 1: "Menengah", 2: "Berisiko"}
    
    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    df_centroids = pd.DataFrame(centroids_original, columns=features)
    
    # Menghitung bobot performa berdasarkan mode parameter pilihan user
    if mode == "Hanya Nilai Ujian (Mid & Final)":
        df_centroids['academic_score'] = df_centroids['midterm_marks'] + df_centroids['final_marks']
    else:
        df_centroids['academic_score'] = (
            df_centroids['final_marks'] + 
            df_centroids['midterm_marks'] + 
            (df_centroids['quiz1_marks'] + df_centroids['quiz2_marks'] + df_centroids['quiz3_marks'])
        )
    
    sorted_clusters = df_centroids['academic_score'].sort_values(ascending=False).index.tolist()
    
    mapping = {}
    if len(sorted_clusters) >= 3:
        mapping[sorted_clusters[0]] = "Tinggi"
        mapping[sorted_clusters[1]] = "Menengah"
        mapping[sorted_clusters[2]] = "Berisiko"
    else:
        for idx, c_id in enumerate(sorted_clusters):
            mapping[c_id] = "Tinggi" if idx == 0 else "Berisiko"
            
    return mapping

def get_cluster_static_info(category_name):
    if category_name == "Tinggi":
        return {
            "label": "🚀 High Achiever (Performa Tinggi)",
            "desc": "Mahasiswa menunjukkan hasil evaluasi parameter terpilih yang sangat memuaskan di atas rata-rata kelas.",
            "saran": "Pertahankan ritme belajar saat ini. Direkomendasikan menjadi asisten dosen atau mentor sebaya.",
            "color": "#2ecc71"
        }
    elif category_name == "Menengah":
        return {
            "label": "📊 Steady / Average (Performa Menengah)",
            "desc": "Mahasiswa menunjukkan hasil evaluasi parameter terpilih yang cukup stabil di tingkat rata-rata kelompok kelas.",
            "saran": "Fokus meningkatkan pemahaman materi inti perkuliahan untuk mendongkrak pencapaian berikutnya.",
            "color": "#f1c40f"
        }
    else: # Berisiko
        return {
            "label": "⚠️ Underperformer (Performa Berisiko)",
            "desc": "Hasil evaluasi parameter kritis berada di tingkat rendah. Risiko tinggi mengalami kendala kelulusan studi.",
            "saran": "Segera jadwalkan sesi bimbingan konseling akademik intensif bersama dosen wali untuk pemulihan nilai.",
            "color": "#e74c3c"
        }

# 5. SIDEBAR NAVIGATION & FITUR BARU: PILIH PARAMETER
with st.sidebar:
    st.title("🎓 Smart Campus AI")
    st.markdown(f"User: **Stevanus**\n\nTema: **Academic Clustering**")
    st.divider()
    
    # REKAYASA FITUR BARU BARU DI SINI:
    st.subheader("⚙️ Konfigurasi Model")
    parameter_mode = st.selectbox(
        "Pilih Parameter Analisis:",
        ["Hanya Nilai Ujian (Mid & Final)", "Semua Parameter Akademik Terintegrasi"]
    )
    
    st.divider()
    menu = st.radio("Navigasi Utama", ["🏠 Dashboard Analisis", "🔍 Prediksi Individu"])
    st.divider()
    uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")

# 6. LOGIKA UTAMA APLIKASI
if model is None or scaler is None:
    st.error("❌ File model (.pkl) tidak ditemukan. Pastikan sudah menjalankan training di Colab!")
else:
    features = ['quiz1_marks', 'quiz2_marks', 'quiz3_marks', 'midterm_marks', 'final_marks', 'previous_gpa', 'lectures_attended', 'labs_attended']
    
    # Jalankan pemetaan otomatis sesuai pilihan parameter di sidebar
    cluster_mapping = get_model_cluster_mapping(model, scaler, features, parameter_mode)

    # --- MENU 1: DASHBOARD ANALISIS MASSAL ---
    if menu == "🏠 Dashboard Analisis":
        st.title("📊 Dashboard Analisis Performa Mahasiswa")
        st.info(f"💡 Mode Analisis Aktif: **{parameter_mode}**")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Preprocessing & Clustering data massal
            df_clean = df[features].fillna(df[features].mean())
            scaled_data = scaler.transform(df_clean)
            df['Cluster'] = model.predict(scaled_data)
            
            df['Kategori_Evaluasi'] = df['Cluster'].map(cluster_mapping)
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

    # --- MENU 2: PREDIKSI INDIVIDU BARU ---
    else:
        st.title("🔍 Prediksi Kategori Mahasiswa")
        st.info(f"💡 Aturan Evaluasi Berdasarkan: **{parameter_mode}**")
        st.write("Masukkan parameter akademis mahasiswa di bawah ini:")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                q1 = st.slider("Quiz 1", 0.0, 10.0, 5.0)
                q2 = st.slider("Quiz 2", 0.0, 10.0, 5.0)
                q3 = st.slider("Quiz 3", 0.0, 10.0, 5.0)
                mid = st.number_input("Midterm Marks (0-30)", 0, 30, 0)
            with c2:
                fin = st.number_input("Final Marks (0-50)", 0, 50, 0)
                gpa = st.slider("Previous GPA (0.0-4.0)", 0.0, 4.0, 2.0)
                lec = st.number_input("Lectures Attended (0-12)", 0, 12, 6)
                lab = st.number_input("Labs Attended (0-6)", 0, 6, 3)
            
            submit = st.form_submit_button("🚀 Analisis Performa")

        if submit:
            # Mengemas data input ke bentuk DataFrame
            input_df = pd.DataFrame([[q1, q2, q3, mid, fin, gpa, lec, lab]], columns=features)
            scaled_input = scaler.transform(input_df)
            predicted_cluster = model.predict(scaled_input)[0]
            
            # LOGIKA ADAPTIF UNTUK INPUT NILAI EKSTREM (Quiz 0 & Final 0)
            if parameter_mode == "Hanya Nilai Ujian (Mid & Final)":
                # Proteksi khusus jika hanya mengukur Mid & Final
                if fin == 0 and mid == 0:
                    assigned_category = "Berisiko"
                else:
                    assigned_category = cluster_mapping.get(predicted_cluster, "Menengah")
            else:
                # Proteksi jika mengukur semua parameter gabungan
                total_skor = fin + mid + q1 + q2 + q3
                if total_skor <= 45.0 or (fin == 0 and mid == 0):
                    assigned_category = "Berisiko"
                else:
                    assigned_category = cluster_mapping.get(predicted_cluster, "Menengah")
            
            info = get_cluster_static_info(assigned_category)
            
            # CETAK KARTU HASIL
            st.markdown(f"""
                <div style="background-color:{info['color']}; padding:30px; border-radius:15px; text-align:center; color:white;">
                    <h1 style="color:white; margin:0;">{info['label']}</h1>
                    <p style="font-size:1.2em; margin:15px 0;">{info['desc']}</p>
                    <div style="background-color:rgba(0,0,0,0.15); padding:15px; border-radius:10px;">
                        <b>💡 Rekomendasi Sistem:</b> {info['saran']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# 7. FOOTER APLIKASI
st.divider()
st.caption(f"UAS Pemrograman AI - Stevanus - Sistem Klaster Komparatif Dinamis")
