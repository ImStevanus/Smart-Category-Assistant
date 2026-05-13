import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_gsheets import GSheetsConnection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Smart Category Assistant Pro", page_icon="🤖", layout="wide")

# URL Google Sheets Anda
SQL_URL = "https://docs.google.com/spreadsheets/d/1EzAuFcdhr77yDHsO2jwGvYt7wmYBbfAB41LAyOd7nFc/edit?usp=sharing" # <--- GANTI INI

# --- KONEKSI DATABASE ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data_from_gsheets():
    # Mengambil data dari Google Sheets
    df = conn.read(spreadsheet=SQL_URL, ttl="0") # ttl="0" agar selalu ambil data terbaru
    df = df.loc[:, ~df.columns.duplicated()]
    cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    df[cols] = df[cols].fillna('')
    df['combined_features'] = df[cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return df

# --- ENGINE AI ---
@st.cache_resource
def train_model(df, k):
    if len(df) < k: k = max(1, len(df))
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)
    return df, vectorizer, model, tfidf_matrix

# --- MAIN APP ---
try:
    df_raw = load_data_from_gsheets()

    with st.sidebar:
        st.title("🤖 Cloud Settings")
        if 'k_val' not in st.session_state: st.session_state['k_val'] = 10
        
        if not df_raw.empty and len(df_raw) > 2:
            if st.button("🔍 Hitung K Ideal"):
                with st.spinner("Menganalisis Cloud Data..."):
                    best_k, max_score = 2, -1
                    limit_k = min(15, len(df_raw) - 1)
                    vec_t = TfidfVectorizer(stop_words='english')
                    mtx_t = vec_t.fit_transform(df_raw['combined_features'])
                    for kt in range(2, limit_k + 1):
                        km = KMeans(n_clusters=kt, random_state=42, n_init=5)
                        labels = km.fit_predict(mtx_t)
                        score = silhouette_score(mtx_t, labels)
                        if score > max_score: max_score, best_k = score, kt
                    st.session_state['k_val'] = best_k
                    st.success(f"Saran K: {best_k}")

        k_val = st.slider("Jumlah Kelompok", 2, 20, st.session_state['k_val'])
        if st.button("🔄 Sinkronisasi Ulang"):
            st.cache_resource.clear()
            st.rerun()

    if df_raw.empty:
        st.warning("Database Google Sheets Kosong!")
        df, vec, model, matrix = None, None, None, None
    else:
        df, vec, model, matrix = train_model(df_raw, k_val)

    st.title("🚀 Smart Category Assistant (Cloud)")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analitik", "➕ Tambah Produk", "🔍 Eksplorasi", "📁 Google Sheets"])

    with tab1:
        if df is not None:
            m1, m2 = st.columns(2)
            m1.metric("Total Data di Cloud", len(df))
            m2.metric("Kelompok AI", k_val)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(matrix.toarray())
            df['x'], df['y'] = coords[:, 0], coords[:, 1]
            fig = px.scatter(df, x='x', y='y', color='Cluster', hover_data=['Product Name'], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("➕ Tambah ke Google Sheets")
        with st.form("gsheets_form", clear_on_submit=True):
            f_name = st.text_input("Nama Produk")
            f_cat = st.text_input("Kategori")
            f_tag = st.text_area("Tag (pisahkan dengan spasi)")
            submitted = st.form_submit_button("Simpan ke Cloud")
            
        if submitted and f_name:
            tags = (f_tag.split() + [""] * 5)[:5]
            new_data = pd.DataFrame([[f_name, f_cat] + tags], 
                                    columns=['Product Name', 'Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
            
            # Gabungkan data lama dan baru
            updated_df = pd.concat([df_raw.drop(columns=['combined_features'], errors='ignore'), new_data], ignore_index=True)
            
            # Update ke Google Sheets
            conn.update(spreadsheet=SQL_URL, data=updated_df)
            st.success("✅ Berhasil dikirim ke Google Sheets!")
            st.cache_resource.clear()
            st.rerun()

    with tab3:
        if df is not None:
            sel = st.selectbox("Cluster:", sorted(df['Cluster'].unique()))
            st.table(df[df['Cluster'] == sel][['Product Name', 'Category']].head(10))

    with tab4:
        st.subheader("📁 Preview Data Live")
        st.dataframe(df_raw.drop(columns=['combined_features'], errors='ignore'), use_container_width=True)

except Exception as e:
    st.error(f"Koneksi GSheets Gagal: {e}")
