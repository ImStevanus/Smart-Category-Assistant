import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. SETTING HALAMAN
st.set_page_config(page_title="Academic Performance Cluster Analyzer", layout="wide")

# 2. LOAD MODEL DAN SCALER
# Pastikan file .pkl hasil dari Colab sudah berada di folder yang sama
try:
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Model atau Scaler tidak ditemukan! Jalankan notebook di Colab dulu untuk menyimpannya.")

# 3. HEADER
st.title("🎓 Academic Performance Cluster Analyzer")
st.markdown("""
Aplikasi ini mengelompokkan mahasiswa ke dalam kategori performa berdasarkan nilai dan tingkat kehadiran 
menggunakan algoritma **K-Means Clustering**.
""")

# 4. INPUT DATA (SIDEBAR)
st.sidebar.header("Input Data Mahasiswa")

def user_input_features():
    quiz1 = st.sidebar.number_input("Nilai Quiz 1", 0.0, 10.0, 7.5)
    quiz2 = st.sidebar.number_input("Nilai Quiz 2", 0.0, 10.0, 7.5)
    quiz3 = st.sidebar.number_input("Nilai Quiz 3", 0.0, 10.0, 7.5)
    midterm = st.sidebar.number_input("Nilai Midterm", 0.0, 100.0, 75.0)
    final = st.sidebar.number_input("Nilai Final", 0.0, 100.0, 80.0)
    prev_gpa = st.sidebar.number_input("IPK Sebelumnya", 0.0, 4.0, 3.0)
    lectures = st.sidebar.number_input("Kehadiran Kuliah (Jam)", 0, 20, 15)
    labs = st.sidebar.number_input("Kehadiran Lab (Jam)", 0, 10, 8)
    
    data = {
        'quiz1_marks': quiz1,
        'quiz2_marks': quiz2,
        'quiz3_marks': quiz3,
        'midterm_marks': midterm,
        'final_marks': final,
        'previous_gpa': prev_gpa,
        'lectures_attended': lectures,
        'labs_attended': labs
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 5. TAMPILKAN INPUT
st.subheader("Data yang Dimasukkan:")
st.write(input_df)

# 6. PROSES PREDIKSI
if st.button("Analisis Cluster Mahasiswa"):
    # Normalisasi data input
    scaled_input = scaler.transform(input_df)
    
    # Prediksi Cluster
    prediction = model.predict(scaled_input)[0]
    
    # Menampilkan Hasil
    st.subheader("Hasil Analisis:")
    
    # Memberi keterangan berdasarkan nomor cluster (Sesuaikan dengan analisis di Colab)
    # Contoh interpretasi:
    if prediction == 0:
        st.success(f"Mahasiswa masuk ke dalam **Cluster {prediction}: Performa Tinggi**")
        st.info("Saran: Pertahankan performa dan bantu teman sejawat.")
    elif prediction == 1:
        st.warning(f"Mahasiswa masuk ke dalam **Cluster {prediction}: Performa Menengah**")
        st.info("Saran: Tingkatkan kehadiran dan fokus pada evaluasi tengah semester.")
    else:
        st.error(f"Mahasiswa masuk ke dalam **Cluster {prediction}: Performa Berisiko**")
        st.info("Saran: Segera lakukan konsultasi akademik dan perbaiki nilai tugas.")

# 7. FOOTER
st.markdown("---")
st.caption("Dibuat untuk UAS Pemrograman AI - Stevanus")
