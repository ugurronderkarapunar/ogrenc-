import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier

# --- 1. ÖZEL FEATURE ENGINEER SINIFI ---
class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        d = X.copy()
        d['ortalama_akademik_puan'] = d[['vize_puani', 'quiz_puani', 'odev_puani']].mean(axis=1, skipna=False)
        d['calisma_uyku_orani']     = d['gunluk_calisma_saati'] / (d['gunluk_uyku_saati'] + 1)
        d['basari_devam_skoru']     = d['onceki_ortalama'] * (d['devam_yuzdesi'] / 100)
        d['toplam_zaman_yuku']      = d[['gunluk_calisma_saati', 'ders_disi_aktivite_saati']].sum(axis=1, skipna=False)
        d['akademik_risk_skoru']    = d['alttan_ders_sayisi'] / d['onceki_ortalama'].replace(0, 0.1)
        return d.replace([np.inf, -np.inf], np.nan)

# --- 2. STREAMLIT SAYFA YAPISI ---
st.set_page_config(
    page_title="Öğrenci Başarı ve Risk Tahmin Sistemi",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Öğrenci Başarı ve Risk Tahmin Paneli")
st.markdown("Bu uygulama, yüklenen Excel veri setini analiz ederek Soft Voting modelini eğitir ve **Geçme / Kalma** durumunu tahmin eder.")

# --- 3. DOSYA YÜKLEME VE MODEL EĞİTME BÖLÜMÜ ---
st.sidebar.header("📁 Veri Seti Yükleme")
uploaded_file = st.sidebar.file_uploader("Excel veri setinizi yükleyin (.xlsx)", type=["xlsx", "xls"])

@st.cache_resource
def modeli_egit(file_source):
    with st.spinner("🔄 Excel dosyası okunuyor ve model eğitiliyor, lütfen bekleyin..."):
        try:
            # Bellek imlecini başa alıp doğrudan Excel olarak oku
            if hasattr(file_source, 'seek'):
                file_source.seek(0)
            df = pd.read_excel(file_source, engine='openpyxl')
        except Exception as e:
            st.error(f"Excel dosyası okunurken hata oluştu! Lütfen geçerli bir .xlsx dosyası yükleyin. Hata detayı: {e}")
            st.stop()
        
        LEAKAGE_COLS = ['final_puani', 'genel_basari_yuzdesi', 'harf_notu']
        df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
        
        HEDEF = 'gecme_kalma_durumu'
        df[HEDEF] = df[HEDEF].map({'Geçti': 0, 'Kaldı': 1})
        df = df.drop_duplicates().reset_index(drop=True)
        
        X = df.drop(columns=HEDEF)
        y = df[HEDEF]
        
        binary_cols = [col for col in X.columns if X[col].nunique() == 2]
        kategorik_cols = [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype) == 'category']
        sayisal_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) and col not in binary_cols]
        FE_COLS = ['ortalama_akademik_puan', 'calisma_uyku_orani', 'basari_devam_skoru', 'toplam_zaman_yuku', 'akademik_risk_skoru']
        
        sayisal_tum = sayisal_cols + FE_COLS
        
        sayisal_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        binary_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))])
        kategorik_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

        prep = ColumnTransformer([
            ('sayisal', sayisal_pipe, sayisal_tum),
            ('binary', binary_pipe, binary_cols),
            ('kategorik', kategorik_pipe, kategorik_cols),
        ], remainder='drop')

        estimators = [
            ('LogReg', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
            ('RandomForest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('HistGB', HistGradientBoostingClassifier(random_state=42)),
            ('ExtraTrees', ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('KNN', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
        ]
        
        pipeline = Pipeline([
            ('fe', StudentFeatureEngineer()),
            ('prep', prep),
            ('model', VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1))
        ])
        
        pipeline.fit(X, y)
        return pipeline, 0.5

# Dosya kontrolü ve yükleme
if uploaded_file is not None:
    model, threshold = modeli_egit(uploaded_file)
    st.sidebar.success("✅ Excel dosyası başarıyla yüklendi ve model hazır!")
else:
    varsayilan_dosya = "duzenlenmis_ogrenci_verisi.xlsx"
    if os.path.exists(varsayilan_dosya):
        model, threshold = modeli_egit(varsayilan_dosya)
    else:
        st.warning("⚠️ Lütfen sol menüden Excel veri setinizi (`duzenlenmis_ogrenci_verisi.xlsx`) yükleyin!")
        st.stop()

# --- 4. KULLANICI GİRDİLERİ (SIDEBAR & FORM) ---
st.sidebar.header("📊 Öğrenci Bilgileri")

vize_puani = st.sidebar.slider("Vize Puanı", 0.0, 100.0, 50.0)
quiz_puani = st.sidebar.slider("Quiz Puanı", 0.0, 100.0, 50.0)
odev_puani = st.sidebar.slider("Ödev Puanı", 0.0, 100.0, 50.0)
onceki_ortalama = st.sidebar.slider("Önceki Dönem Ortalaması", 0.0, 4.0, 2.5)
devam_yuzdesi = st.sidebar.slider("Devam Yüzdesi (%)", 0.0, 100.0, 80.0)
gunluk_calisma_saati = st.sidebar.slider("Günlük Çalışma Saati", 0.0, 12.0, 3.0)
gunluk_uyku_saati = st.sidebar.slider("Günlük Uyku Saati", 0.0, 12.0, 7.0)
ders_disi_aktivite_saati = st.sidebar.slider("Ders Dışı Aktivite Saati", 0.0, 10.0, 2.0)
alttan_ders_sayisi = st.sidebar.number_input("Alttan Alınan Ders Sayısı", 0, 10, 0)

giris_verisi = {
    'vize_puani': vize_puani,
    'quiz_puani': quiz_puani,
    'odev_puani': odev_puani,
    'onceki_ortalama': onceki_ortalama,
    'devam_yuzdesi': devam_yuzdesi,
    'gunluk_calisma_saati': gunluk_calisma_saati,
    'gunluk_uyku_saati': gunluk_uyku_saati,
    'ders_disi_aktivite_saati': ders_disi_aktivite_saati,
    'alttan_ders_sayisi': alttan_ders_sayisi,
}

# --- 5. TAHMİN ÇALIŞTIRMA ---
if st.button("🚀 Başarı Durumunu Tahmin Et", type="primary"):
    df_input = pd.DataFrame([giris_verisi])
    
    try:
        proba = model.predict_proba(df_input)[:, 1][0]
        karar = 1 if proba >= threshold else 0
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Kalma / Risk Olasılığı", value=f"%{proba * 100:.2f}")
            
        with col2:
            if karar == 1:
                st.error("⚠️ **Riskli Durum (Kalış Bekleniyor)**")
            else:
                st.success("✅ **Başarılı (Geçiş Bekleniyor)**")
                
        st.info(f"💡 Kullanılan Karar Eşik Değeri: **{threshold:.3f}**")
        
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
