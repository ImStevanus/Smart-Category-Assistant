import streamlit as st
import pandas as pd
import plotly.express as px
import os
import csv
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
    # Jika file tidak ada, buat file baru dengan header standar
    if not os.path.exists(file_path):
        df_empty = pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
        df_empty.to_csv(file_path, index=False)
        return df_empty
    
    # on_bad_lines='skip' berfungsi agar app tidak crash jika ada baris CSV yang rusak
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except:
        # Jika file benar-benar rusak parah, buat dataframe minimal
        return pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])

    df = df.loc[:, ~df.columns.duplicated()] # Hapus kolom duplikat
    
    cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    for col in cols:
        if col not in df.columns:
            df[col] = ""
            
    df[cols] = df[cols].fillna('')
    # Gabungkan semua kolom tag menjadi satu fitur teks untuk diproses AI
    df['combined_features'] = df[cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return df

# --- FUNGSI TRAINING MODEL ---
@st.cache_resource
def train_model(df, k):
    # Proteksi jika data lebih sedikit dari jumlah cluster yang diminta
    if len(df) < k:
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
        st.info("Atur jumlah kelompok produk di bawah ini.")
        k_val = st.sidebar.slider("Jumlah Kelompok (K)", 2, 20, 10)
        
        st.divider()
        if st.button("🔄 Latih Ulang Model", help="Klik ini untuk memperbarui cluster setelah menambah data baru"):
            st.cache_resource.clear()
            st.rerun()

    # Cek jika data kosong
    if df_raw.empty:
        st.warning("Database kosong. Silakan tambahkan produk baru di tab 'Input Produk Baru'.")
        df, vec, model, matrix = None, None, None, None
    else:
        # Jalankan Engine Clustering
        df, vec, model, matrix = train_model(df_raw, k_val)

    st.title("🤖 Smart Category Assistant")
    st.markdown("Asisten cerdas untuk manajemen inventaris menggunakan Clustering AI.")

    # Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Analitik", 
        "🔮 Input Produk Baru", 
        "🔍 Eksplorasi Segmen", 
        "📁 Database Master"
    ])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        if df is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Produk", len(df))
            m2.metric("Kelompok Terbentuk", k_val)
            m3.metric("Status Sistem", "Aktif")

            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader("Peta Kedekatan Produk (PCA)")
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
        else:
            st.info("Dashboard akan muncul setelah Anda memasukkan data.")

    # --- TAB 2: INPUT PRODUK BARU (DENGAN PERBAIKAN CSV) ---
    with tab2:
        st.subheader("🔮 Input & Prediksi Otomatis")
        st.write("Tambahkan produk baru ke database. Sistem akan memprediksi cluster-nya.")
        
        with st.form("new_product_form", clear_on_submit=True):
            f_name = st.text_input("Nama Produk")
            f_cat = st.text_input("Kategori Utama")
            f_tags = st.text_area("Masukkan Tag / Deskripsi (pisahkan dengan spasi)")
            submitted = st.form_submit_button("Simpan & Analisis")
            
        if submitted:
            if f_name and f_cat:
                # Membersihkan input agar tidak merusak kolom CSV
                clean_name = f_name.replace(',', ' ')
                clean_cat = f_cat.replace(',', ' ')
                clean_tags = f_tags.replace(',', ' ').split()

                # Buat baris baru dengan format 7 kolom yang konsisten
                new_row = [clean_name, clean_cat] + (clean_tags + [""] * 5)[:5]
                
                # SIMPAN KE CSV (Gunakan QUOTING_ALL agar koma di dalam teks tidak merusak kolom)
                file_path = 'data_set.csv'
                with open(file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    writer.writerow(new_row)

                st.success(f"✅ Produk '{f_name}' berhasil disimpan!")

                # Lakukan Prediksi jika model tersedia
                if vec is not None:
                    input_text = f"{clean_cat} {' '.join(clean_tags)}"
                    input_vec = vec.transform([input_text])
                    prediction = model.predict(input_vec)[0]
                    dist = model.transform(input_vec).min()
                    
                    if dist > 0.85:
                        st.warning(f"⚠️ **Produk Unik!** Karakteristiknya baru. Sementara masuk ke **Cluster {prediction}**.")
                    else:
                        st.info(f"Produk ini masuk ke dalam **Cluster {prediction}**.")
                
                st.info("💡 Tekan tombol 'Latih Ulang Model' di sidebar untuk memperbarui peta cluster.")
            else:
                st.error("Nama Produk dan Kategori wajib diisi!")

    # --- TAB 3: DETAIL SEGMEN ---
    with tab3:
        if df is not None:
            sel_cluster = st.selectbox("Pilih nomor cluster:", sorted(df['Cluster'].unique()))
            cluster_filtered = df[df['Cluster'] == sel_cluster]
            
            c_stats, c_table = st.columns([1, 2])
            with c_stats:
                st.write(f"**Top Kata Kunci Cluster {sel_cluster}**")
                all_words = " ".join(cluster_filtered['combined_features']).split()
                top_words = pd.Series([w for w in all_words if len(w) > 3]).value_counts().head(10)
                st.plotly_chart(px.bar(top_words, orientation='h'), use_container_width=True)
                
            with c_table:
                st.write(f"**Daftar Produk di Cluster {sel_cluster}**")
                st.dataframe(cluster_filtered[['Product Name', 'Category', 'tag1', 'tag2']], use_container_width=True)
        else:
            st.info("Belum ada data untuk dianalisis.")

    # --- TAB 4: MASTER DATA ---
    with tab4:
        st.subheader("Database Inventory")
        if not df_raw.empty:
            final_display = df.drop(columns=['combined_features', 'x', 'y'], errors='ignore') if df is not None else df_raw
            st.dataframe(final_display, use_container_width=True)
            
            csv_data = final_display.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report CSV", csv_data, "inventory_report.csv", "text/csv")
        else:
            st.info("Database masih kosong.")

except Exception as e:
    st.error(f"Sistem mengalami kendala: {e}")
    st.info("Tips: Jika error terus berlanjut, coba hapus file 'data_set.csv' dan biarkan sistem membuatnya kembali.")
