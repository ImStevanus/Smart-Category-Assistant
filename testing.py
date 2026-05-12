import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Smart Category Assistant", page_icon="🤖", layout="wide")

# --- CSS CUSTOM ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 5px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA & MODEL ENGINE ---
@st.cache_resource
def train_model(df, k):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(tfidf_matrix)
    return df, vectorizer, model, tfidf_matrix

@st.cache_data
def get_data():
    df = pd.read_csv('data_set.csv')
    df = df.loc[:, ~df.columns.duplicated()]
    cols = ['Category', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']
    df[cols] = df[cols].fillna('')
    df['combined_features'] = df[cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return df

try:
    df_raw = get_data()
    
    # Sidebar
    st.sidebar.title("🤖 Smart Settings")
    k_val = st.sidebar.slider("Tentukan Jumlah Kelompok (K)", 2, 15, 10)
    
    # Jalankan Engine
    df, vec, model, matrix = train_model(df_raw, k_val)

    st.title("🤖 Smart Category Assistant")
    st.markdown("Asisten cerdas untuk otomatisasi pengelompokan produk dan analisis inventaris.")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Utama", "🎯 Prediksi Produk Baru", "🔍 Detail Cluster", "📁 Master Data"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Produk", len(df))
        c2.metric("Total Segmen", k_val)
        c3.metric("Akurasi Distribusi", "Optimal" if k_val >= 8 else "General")

        col_map, col_dist = st.columns([2, 1])
        with col_map:
            st.subheader("Peta Kedekatan Produk (PCA)")
            pca = PCA(n_components=2)
            coords = pca.fit_transform(matrix.toarray())
            df['x'], df['y'] = coords[:, 0], coords[:, 1]
            fig = px.scatter(df, x='x', y='y', color='Cluster', hover_data=['Product Name'], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with col_dist:
            st.subheader("Populasi per Cluster")
            counts = df['Cluster'].value_counts().reset_index()
            st.plotly_chart(px.bar(counts, x='Cluster', y='count', color='Cluster'), use_container_width=True)

    # --- TAB 2: PREDIKSI (FITUR BARU) ---
    with tab2:
        st.subheader("🔮 Prediksi Kategori untuk Produk Baru")
        st.write("Masukkan detail produk baru di bawah ini untuk melihat ke kelompok mana produk tersebut seharusnya masuk.")
        
        with st.form("predict_form"):
            new_name = st.text_input("Nama Produk Baru (Contoh: Gaming Mouse)")
            new_cat = st.text_input("Kategori Dasar (Contoh: Electronics)")
            new_tags = st.text_area("Masukkan Tag (Pisahkan dengan spasi, contoh: RGB Wireless Gaming)")
            submit = st.form_submit_button("Analisis Cluster")
            
        if submit:
            input_text = f"{new_cat} {new_tags}"
            input_vec = vec.transform([input_text])
            prediction = model.predict(input_vec)[0]
            
            st.success(f"Produk '**{new_name}**' diprediksi masuk ke **Cluster {prediction}**")
            
            # Tampilkan produk serupa di cluster yang sama
            st.write("Produk serupa yang sudah ada di sistem:")
            similar = df[df['Cluster'] == prediction][['Product Name', 'Category']].head(5)
            st.table(similar)

    # --- TAB 3: DETAIL ---
    with tab3:
        sel_c = st.selectbox("Pilih Cluster untuk dibedah:", sorted(df['Cluster'].unique()))
        cdat = df[df['Cluster'] == sel_c]
        
        col_w1, col_w2 = st.columns([1, 2])
        with col_w1:
            st.write(f"**Karakteristik Cluster {sel_c}**")
            words = " ".join(cdat['combined_features']).split()
            top_w = pd.Series([w for w in words if len(w) > 3]).value_counts().head(10)
            st.plotly_chart(px.bar(top_w, orientation='h', title="Top Keywords"), use_container_width=True)
        with col_w2:
            st.write(f"**Daftar Produk (Total: {len(cdat)})**")
            st.dataframe(cdat[['Product Name', 'Category', 'tag1', 'tag2']], use_container_width=True)

    # --- TAB 4: DATA ---
    with tab4:
        st.subheader("Dataset Tersegmentasi")
        st.dataframe(df.drop(columns=['combined_features', 'x', 'y']), use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Data Lengkap", csv, "hasil_segmentasi.csv", "text/csv")

except Exception as e:
    st.error(f"Oops! Terjadi kesalahan: {e}")
