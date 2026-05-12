import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart Category Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- FUNGSI LOAD & PREPROCESS DATA ---
def get_data():
    file_path = 'data_set.csv'
    # Jika file tidak ada, buat file baru dengan header
    if not os.path.exists(file_path):
        df_empty = pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
        df_empty.to_csv(file_path, index=False)
        return df_empty
    
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.duplicated()] # Hapus kolom duplikat
    cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    
    # Pastikan semua kolom fitur ada
    for col in cols:
        if col not in df.columns:
            df[col] = ""
            
    df[cols] = df[cols].fillna('')
    # Gabungkan fitur untuk Vectorizer
    df['combined_features'] = df[cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return df

# --- FUNGSI TRAINING MODEL ---
@st.cache_resource
def train_model(df, k):
    if len(df) < k: # Proteksi jika data lebih sedikit dari jumlah cluster
        k = max(1, len(df))
        
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)
    
    return df, vectorizer, model, tfidf_matrix

# --- MAIN APP ---
try:
    df_raw = get_data()

    # Sidebar: Pengaturan
    with st.sidebar:
        st.title("🤖 Assistant Settings")
        st.info("Atur parameter kecerdasan di sini.")
        k_val = st.sidebar.slider("Jumlah Kelompok (K)", 2, 20, 10)
        
        st.divider()
        if st.button("🔄 Latih Ulang Model", help="Gunakan ini setelah menambah banyak data baru"):
            st.cache_resource.clear()
            st.rerun()

    # Jalankan Engine
    df, vec, model, matrix = train_model(df_raw, k_val)

    st.title("🤖 Smart Category Assistant")
    st.markdown("Sistem otomasi manajemen inventaris berbasis *Unsupervised Learning*.")

    # Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Analitik", 
        "🔮 Input Produk Baru", 
        "🔍 Eksplorasi Segmen", 
        "📁 Database Master"
    ])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Produk", len(df))
        m2.metric("Kelompok Terbentuk", k_val)
        m3.metric("Status Data", "Live & Sync")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Peta Kedekatan Produk")
            pca = PCA(n_components=2)
            coords = pca.fit_transform(matrix.toarray())
            df['x'], df['y'] = coords[:, 0], coords[:, 1]
            fig = px.scatter(df, x='x', y='y', color='Cluster', 
                           hover_data=['Product Name', 'Category'],
                           template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("Populasi Per Kelompok")
            counts = df['Cluster'].value_counts().reset_index()
            counts.columns = ['Cluster', 'Jumlah']
            st.plotly_chart(px.bar(counts, x='Cluster', y='Jumlah', color='Cluster'), use_container_width=True)

    # --- TAB 2: INPUT PRODUK BARU (WITH AUTO-SAVE) ---
    with tab2:
        st.subheader("🔮 Input & Prediksi Otomatis")
        st.write("Tambahkan produk baru. Sistem akan memprediksi kelompoknya dan menyimpan data secara permanen.")
        
        with st.form("new_product_form", clear_on_submit=True):
            f_name = st.text_input("Nama Produk")
            f_cat = st.text_input("Kategori Utama")
            f_tags = st.text_area("Deskripsi Singkat / Tag (pisahkan dengan spasi)")
            submitted = st.form_submit_button("Simpan ke Database & Analisis")
            
        if submitted:
            if f_name and f_cat:
                # 1. Proses untuk Prediksi
                input_text = f"{f_cat} {f_tags}"
                input_vec = vec.transform([input_text])
                prediction = model.predict(input_vec)[0]
                
                # Hitung jarak untuk cek keunikan
                dist = model.transform(input_vec).min()

                # 2. Simpan secara permanen ke CSV
                new_row = [f_name, f_cat] + (f_tags.split() + [""] * 5)[:5]
                new_df = pd.DataFrame([new_row], columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
                new_df.to_csv('data_set.csv', mode='a', index=False, header=False)

                # 3. Tampilkan Hasil
                st.success(f"✅ Produk '{f_name}' berhasil disimpan ke database!")
                
                if dist > 0.85:
                    st.warning(f"⚠️ **Produk Unik Terdeteksi!** Produk ini memiliki karakteristik baru yang belum pernah ada sebelumnya. Sementara dimasukkan ke **Cluster {prediction}**.")
                else:
                    st.info(f"Produk ini sangat cocok masuk ke dalam **Cluster {prediction}**.")
                
                st.info("💡 Klik tombol 'Latih Ulang Model' di sidebar agar produk baru ini ikut membentuk pola cluster yang baru.")
            else:
                st.error("Nama Produk dan Kategori wajib diisi!")

    # --- TAB 3: DETAIL SEGMEN ---
    with tab3:
        sel_cluster = st.selectbox("Pilih nomor cluster:", sorted(df['Cluster'].unique()))
        cluster_filtered = df[df['Cluster'] == sel_cluster]
        
        c_stats, c_table = st.columns([1, 2])
        with c_stats:
            st.write(f"**Kata Kunci Dominan di Cluster {sel_cluster}**")
            all_words = " ".join(cluster_filtered['combined_features']).split()
            top_words = pd.Series([w for w in all_words if len(w) > 3]).value_counts().head(10)
            st.plotly_chart(px.bar(top_words, orientation='h'), use_container_width=True)
            
        with c_table:
            st.write(f"**Daftar Produk di Cluster {sel_cluster}**")
            st.dataframe(cluster_filtered[['Product Name', 'Category', 'tag1', 'tag2']], use_container_width=True)

    # --- TAB 4: MASTER DATA ---
    with tab4:
        st.subheader("Data Inventory Terkini")
        st.write("Berikut adalah seluruh data yang telah digabungkan dengan hasil segmentasi AI.")
        final_display = df.drop(columns=['combined_features', 'x', 'y'], errors='ignore')
        st.dataframe(final_display, use_container_width=True)
        
        csv_data = final_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Laporan CSV", csv_data, "inventory_report.csv", "text/csv")

except Exception as e:
    st.error(f"Sistem mengalami kendala: {e}")
    st.info("Pastikan file 'data_set.csv' berada di folder yang sama dengan file ini.")
