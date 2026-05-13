import streamlit as st
import pandas as pd
import plotly.express as px
import os
from streamlit_gsheets import GSheetsConnection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Smart Category Assistant Hybrid", page_icon="🤖", layout="wide")

# URL Google Sheets (Data Baru)
SQL_URL = "https://docs.google.com/spreadsheets/d/1EzAuFcdhr77yDHsO2jwGvYt7wmYBbfAB41LAyOd7nFc/edit?usp=sharing"

# --- KONEKSI ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_hybrid_data():
    # 1. Load Data Awal dari CSV Lokal
    if os.path.exists('data_set.csv'):
        df_csv = pd.read_csv('data_set.csv', on_bad_lines='skip', engine='python')
    else:
        df_csv = pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])

    # 2. Load Data Baru dari Google Sheets
    try:
        df_gsheets = conn.read(spreadsheet=SQL_URL, ttl="0")
    except:
        df_gsheets = pd.DataFrame(columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])

    # 3. Gabungkan Keduanya
    df_combined = pd.concat([df_csv, df_gsheets], ignore_index=True)
    
    # Bersihkan kolom duplikat
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    
    # Kolom yang wajib ada
    cols_needed = ['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    for col in cols_needed:
        if col not in df_combined.columns:
            df_combined[col] = ""
    
    # --- FIX ERROR: Konversi semua ke string dan tangani NaN ---
    df_combined[cols_needed] = df_combined[cols_needed].fillna('').astype(str)
    
    # Menghapus string 'nan' yang sering muncul saat konversi float ke str
    df_combined[cols_needed] = df_combined[cols_needed].replace(['nan', 'None', '<NA>'], '')

    # Gabungkan fitur untuk AI dengan proteksi tipe data
    feature_cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    # Menggunakan filter(None, ...) untuk membuang string kosong saat penggabungan
    df_combined['combined_features'] = [
        ' '.join(filter(None, [str(val) for val in row])) 
        for row in df_combined[feature_cols].values
    ]
    
    return df_combined, df_gsheets

# --- ENGINE AI ---
@st.cache_resource
def train_model(df, k):
    if len(df) < k: k = max(1, len(df))
    # Vectorizer akan mengubah teks menjadi matriks angka
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)
    
    return df, vectorizer, model, tfidf_matrix

# --- MAIN APP ---
try:
    df_full, df_only_gsheets = load_hybrid_data()

    with st.sidebar:
        st.title("🤖 Hybrid Assistant")
        if 'k_val' not in st.session_state: st.session_state['k_val'] = 10
        
        if not df_full.empty and len(df_full) > 2:
            if st.button("🔍 Cari K Ideal"):
                with st.spinner("Menganalisis..."):
                    best_k, max_score = 2, -1
                    limit_k = min(15, len(df_full) - 1)
                    vec_t = TfidfVectorizer(stop_words='english')
                    mtx_t = vec_t.fit_transform(df_full['combined_features'])
                    for kt in range(2, limit_k + 1):
                        km = KMeans(n_clusters=kt, random_state=42, n_init=5)
                        labels = km.fit_predict(mtx_t)
                        score = silhouette_score(mtx_t, labels)
                        if score > max_score: max_score, best_k = score, kt
                    st.session_state['k_val'] = best_k
                    st.success(f"Saran K: {best_k}")

        k_val = st.slider("Jumlah Kelompok", 2, 20, st.session_state['k_val'])
        if st.button("🔄 Refresh Data & AI"):
            st.cache_resource.clear()
            st.rerun()

    if df_full.empty:
        st.warning("Database kosong! Pastikan file CSV atau Google Sheets tersedia.")
    else:
        df, vec, model, matrix = train_model(df_full, k_val)

        st.title("🚀 Smart Category Assistant (Hybrid)")
        st.caption("Mode: CSV (Lokal) + Google Sheets (Cloud)")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analitik", "➕ Tambah Produk", "🔍 Eksplorasi", "📁 Master Data"])

        with tab1:
            c1, c2 = st.columns(2)
            c1.metric("Total Produk", len(df))
            c2.metric("Kelompok AI", k_val)
            
            # Visualisasi Kedekatan Produk
            pca = PCA(n_components=2)
            coords = pca.fit_transform(matrix.toarray())
            df['x'], df['y'] = coords[:, 0], coords[:, 1]
            fig = px.scatter(df, x='x', y='y', color='Cluster', 
                           hover_data=['Product Name', 'Category'], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("➕ Simpan ke Google Sheets")
            with st.form("input_form", clear_on_submit=True):
                f_name = st.text_input("Nama Produk")
                f_cat = st.text_input("Kategori")
                f_tag = st.text_area("Tag")
                submitted = st.form_submit_button("Simpan Permanen")
            
            if submitted and f_name:
                tags_list = (f_tag.split() + [""] * 5)[:5]
                new_row = pd.DataFrame([[f_name, f_cat] + tags_list], 
                                        columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
                
                # Pastikan data gsheets lama dibersihkan dari kolom AI sebelum digabung
                clean_gsheets = df_only_gsheets.copy()
                updated_gsheets = pd.concat([clean_gsheets, new_row], ignore_index=True)
                
                conn.update(spreadsheet=SQL_URL, data=updated_gsheets)
                st.success(f"✅ '{f_name}' berhasil disimpan ke Cloud!")
                st.cache_resource.clear()
                st.rerun()

        with tab3:
            sel_c = st.selectbox("Pilih Cluster:", sorted(df['Cluster'].unique()))
            st.dataframe(df[df['Cluster'] == sel_c][['Product Name', 'Category', 'tag1', 'tag2']], use_container_width=True)

        with tab4:
            st.subheader("📁 Database Master (Gabungan)")
            st.dataframe(df.drop(columns=['combined_features', 'x', 'y', 'Cluster']), use_container_width=True)

except Exception as e:
    st.error(f"Sistem Error: {e}")
