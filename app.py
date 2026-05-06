import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Shopper Segment AI", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI HELPER ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def preprocess_data(df):
    df_clean = df.copy()
    # Mengubah kolom kategorikal menjadi angka
    le = LabelEncoder()
    cat_cols = ['Month', 'VisitorType', 'Weekend', 'Revenue']
    for col in cat_cols:
        df_clean[col] = le.fit_transform(df_clean[col])
    return df_clean

# --- SIDEBAR: INPUT DATA ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
st.sidebar.title("Kontrol Sistem")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

# --- MAIN CONTENT ---
st.title("🛍️ Online Shoppers Segment Analyzer")
st.markdown("Sistem Cerdas Analisis Segmentasi Pelanggan untuk Optimasi Strategi Marketing.")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    df_numeric = preprocess_data(data)

    # Tab menu
    tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data", "🤖 Training AI", "💡 Insight Bisnis"])

    with tab1:
        st.subheader("Preview Dataset")
        st.dataframe(data.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data", f"{data.shape[0]} baris")
        col2.metric("Total Fitur", f"{data.shape[1]} kolom")
        col3.metric("Rata-rata PageValue", f"{data['PageValues'].mean():.2f}")

        st.subheader("Distribusi Niat Beli (Revenue)")
        fig_rev = px.pie(data, names='Revenue', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_rev)

    with tab2:
        st.subheader("Konfigurasi Model K-Means")
        
        # Pemilihan Fitur
        all_features = df_numeric.columns.tolist()
        selected_features = st.multiselect(
            "Pilih fitur untuk segmentasi:", 
            all_features, 
            default=['Administrative_Duration', 'ProductRelated_Duration', 'BounceRates', 'PageValues']
        )

        if len(selected_features) >= 2:
            X = df_numeric[selected_features]
            
            # Scalling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Method Otomatis
            st.write("🔍 Mencari Jumlah Cluster Optimal (Elbow Method)...")
            distortions = []
            K_range = range(1, 11)
            for k in K_range:
                kmeanModel = KMeans(n_clusters=k, n_init=10, random_state=42)
                kmeanModel.fit(X_scaled)
                distortions.append(kmeanModel.inertia_)

            fig_elbow = px.line(x=list(K_range), y=distortions, markers=True, title="Elbow Method")
            fig_elbow.update_layout(xaxis_title="Jumlah Cluster (k)", yaxis_title="Inertia")
            st.plotly_chart(fig_elbow, use_container_width=True)

            k_input = st.slider("Tentukan Jumlah Cluster:", 2, 6, 3)

            # Proses Clustering
            model = KMeans(n_clusters=k_input, n_init=10, random_state=42)
            data['Cluster'] = model.fit_predict(X_scaled)

            st.success(f"Model berhasil dilatih dengan {k_input} cluster!")

            # Visualisasi Cluster
            st.subheader("Visualisasi Sebaran Segmen")
            fig_cluster = px.scatter(
                data, 
                x=selected_features[0], 
                y=selected_features[1], 
                color='Cluster',
                hover_data=['VisitorType', 'Month', 'Revenue'],
                title=f"Cluster berdasarkan {selected_features[0]} & {selected_features[1]}"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("Pilih minimal 2 fitur untuk memulai clustering.")

    with tab3:
        if 'Cluster' in data.columns:
            st.subheader("Analisis Karakteristik Per Segmen")
            
            # Hitung rata-rata per cluster
            cluster_analysis = data.groupby('Cluster')[selected_features].mean()
            st.write(cluster_analysis)

            # Penjelasan Cerdas (Interpretasi)
            st.markdown("### 📝 Interpretasi Hasil")
            for i in range(k_input):
                with st.expander(f"Analisis Segmen {i}"):
                    row = cluster_analysis.iloc[i]
                    if row['PageValues'] > data['PageValues'].mean():
                        st.write("**Profil:** High Value Customer")
                        st.write("**Saran:** Berikan program loyalitas atau preview produk eksklusif.")
                    elif row['BounceRates'] > data['BounceRates'].mean():
                        st.write("**Profil:** Uninterested/Window Shopper")
                        st.write("**Saran:** Tingkatkan UI/UX atau berikan pop-up diskon instan.")
                    else:
                        st.write("**Profil:** Regular Visitor")
                        st.write("**Saran:** Kirimkan newsletter berkala melalui email marketing.")
        else:
            st.info("Selesaikan training di Tab 'Training AI' terlebih dahulu.")

else:
    # Tampilan awal jika belum upload file
    st.image("https://images.unsplash.com/photo-1551288049-bbbda536339a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_container_width=True)
    st.info("Silakan upload file 'online_shoppers_intention.csv' melalui sidebar untuk memulai analisis.")

# --- FOOTER ---
st.divider()
st.caption("UAS Sistem Cerdas - Online Shoppers Segment Analyzer - 2024")
