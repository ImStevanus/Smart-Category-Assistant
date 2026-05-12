import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px # Untuk grafik interaktif

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Product Segmenter Pro",
    page_icon="📊",
    layout="wide"
)

# --- CUSTOM CSS UNTUK UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_process():
    df = pd.read_csv('data_set.csv')
    tag_cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    df[tag_cols] = df[tag_cols].fillna('')
    df['combined_features'] = df[tag_cols].apply(lambda x: ' '.join(x), axis=1)
    return df, tag_cols

try:
    df_raw, tag_cols = load_and_process()
    df = df_raw.copy()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Kontrol Panel")
        st.info("Sesuaikan parameter clustering di sini.")
        k_value = st.slider("Jumlah Cluster (k)", 2, 20, 10)
        
        st.divider()
        st.subheader("Tentang Sistem")
        st.caption("Sistem ini menggunakan TF-IDF Vectorization dan K-Means untuk pemetaan produk otomatis.")

    # --- ENGINE CLUSTERING ---
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)

    # --- HEADER ---
    st.title("📊 Product Segmentation Dashboard")
    st.write(f"Menampilkan analisis untuk **{len(df)}** produk unik.")

    # --- TABS UI ---
    tab1, tab2, tab3 = st.tabs(["📈 Analisis Cluster", "🔍 Pencarian & Filter", "📄 Data Master"])

    with tab1:
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Produk", len(df))
        m2.metric("Jumlah Cluster", k_value)
        m3.metric("Rata-rata Produk/Cluster", int(len(df)/k_value))

        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            st.subheader("Visualisasi Ruang Produk (PCA 2D)")
            pca = PCA(n_components=2)
            coords = pca.fit_transform(tfidf_matrix.toarray())
            df_viz = df.copy()
            df_viz['x'] = coords[:, 0]
            df_viz['y'] = coords[:, 1]
            
            # Menggunakan Plotly agar grafik bisa di-zoom dan interaktif
            fig_pca = px.scatter(
                df_viz, x='x', y='y', color='Cluster',
                hover_data=['Product Name', 'Category'],
                template="plotly_white",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with col_right:
            st.subheader("Distribusi Cluster")
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', color='Count', color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("Eksplorasi Detail Cluster")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            sel_cluster = st.selectbox("Pilih Cluster:", sorted(df['Cluster'].unique()))
            cluster_data = df[df['Cluster'] == sel_cluster]
            
            # Fitur Baru: Top Keywords dalam Cluster
            st.write("**Karakteristik Cluster:**")
            words = " ".join(cluster_data['combined_features']).split()
            top_words = pd.Series([w for w in words if len(w) > 3]).value_counts().head(10)
            st.dataframe(top_words, column_config={"index": "Kata", "0": "Frekuensi"})

        with c2:
            st.write(f"Daftar Produk di Cluster {sel_cluster}")
            st.dataframe(cluster_data[['Product Name', 'Category'] + tag_cols[:3]], use_container_width=True)

        st.divider()
        st.subheader("🔎 Cari Produk")
        search_query = st.text_input("Masukkan nama produk untuk mengecek clusternya:")
        if search_query:
            results = df[df['Product Name'].str.contains(search_query, case=False)]
            if not results.empty:
                st.success(f"Ditemukan {len(results)} produk.")
                st.table(results[['Product Name', 'Category', 'Cluster']])
            else:
                st.warning("Produk tidak ditemukan.")

    with tab3:
        st.subheader("Seluruh Data Tersegmentasi")
        st.dataframe(df.drop(columns=['combined_features']), use_container_width=True)
        
        # Download Section
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report (CSV)", data=csv, file_name='report_segmentasi.csv', mime='text/csv')

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.info("Pastikan file 'data_set.csv' tersedia di direktori yang sama.")
