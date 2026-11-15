import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Öğrenci Başarı ve Risk Tahmin Sistemi",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Öğrenci Başarı ve Risk Tahmin Asistanı")
st.markdown("Aşağıdaki akademik ve kişisel bilgileri girerek öğrencinin **Geçme / Kalma** durumunu anlık tahmin edin.")
st.markdown("---")

# --- ÖZEL FEATURE ENGINEER SINIFI ---
class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        d['ortalama_akademik_puan'] = d[['vize_puani', 'quiz_puani', 'odev_puani']].mean(axis=1, skipna=True)
        d['calisma_uyku_orani']     = d['gunluk_calisma_saati'] / (d['gunluk_uyku_saati'] + 1)
        d['basari_devam_skoru']     = d['onceki_ortalama'] * (d['devam_yuzdesi'] / 100)
        d['toplam_zaman_yuku']      = d[['gunluk_calisma_saati', 'ders_disi_aktivite_saati']].sum(axis=1, skipna=True)
        d['akademik_risk_skoru']    = d['alttan_ders_sayisi'] / d['onceki_ortalama'].replace(0, 0.1)
        return d.replace([np.inf, -np.inf], np.nan)

# --- MODELİ VE VERİYİ YÜKLEME / EĞİTME (GÜVENLİ VE OTOMATİK) ---
@st.cache_resource
def modeli_hazirla_ve_yukle():
    df = None
    tum_adaylar = glob.glob("**/*.xlsx", recursive=True) + glob.glob("**/*.csv", recursive=True)
    for dosya in tum_adaylar:
        if "sample_data" in dosya: 
            continue
        try:
            if dosya.endswith('.xlsx'):
                df = pd.read_excel(dosya)
            else:
                for enc in ['utf-8', 'cp1254', 'latin1']:
                    try:
                        df = pd.read_csv(dosya, encoding=enc)
                        break
                    except:
                        continue
            if df is not None and not df.empty:
                break
        except:
            continue

    HEDEF = 'gecme_kalma_durumu'
    
    # Eğer gerçek veri dosyası bulunamadıysa veya hedef sütun yoksa sentetik veri üret
    if df is None or df.empty or HEDEF not in df.columns:
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'vize_puani': np.random.randint(20, 100, n_samples),
            'quiz_puani': np.random.randint(10, 100, n_samples),
            'odev_puani': np.random.randint(30, 100, n_samples),
            'gunluk_calisma_saati': np.random.uniform(1, 8, n_samples),
            'gunluk_uyku_saati': np.random.uniform(4, 10, n_samples),
            'onceki_ortalama': np.random.uniform(1.5, 4.0, n_samples),
            'devam_yuzdesi': np.random.randint(50, 100, n_samples),
            'ders_disi_aktivite_saati': np.random.randint(0, 5, n_samples),
            'alttan_ders_sayisi': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            'cinsiyet': np.random.choice(['K', 'E'], n_samples),
            'internet_erisimi': np.random.choice(['Evet', 'Hayir'], n_samples),
            'gecme_kalma_durumu': np.random.choice(['Geçti', 'Kaldı'], n_samples, p=[0.75, 0.25])
        })

    LEAKAGE_COLS = ['final_puani', 'genel_basari_yuzdesi', 'harf_notu']
    id_cols = [c for c in df.columns if 'id' in c.lower() or ('no' in c.lower() and c != HEDEF)]
    if id_cols:
        df = df.drop(columns=id_cols)
    
    existing_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    if existing_leakage:
        df = df.drop(columns=existing_leakage)
    
    # Hedef sütunu güvenli şekilde sayısal (0 ve 1) formata dönüştür
    if df[HEDEF].dtype == 'object' or str(df[HEDEF].dtype).name == 'category':
        df[HEDEF] = df[HEDEF].map({'Geçti': 0, 'Kaldı': 1})
    
    df[HEDEF] = pd.to_numeric(df[HEDEF], errors='coerce')
    df = df.dropna(subset=[HEDEF])

    X = df.drop(columns=HEDEF)
    y = df[HEDEF].astype(int)

    binary_cols, kategorik_cols, sayisal_cols = [], [], []
    for col in X.columns:
        if X[col].nunique() == 2:
            binary_cols.append(col)
        elif X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            kategorik_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            sayisal_cols.append(col)

    FE_COLS = ['ortalama_akademik_puan', 'calisma_uyku_orani', 'basari_devam_skoru',
               'toplam_zaman_yuku', 'akademik_risk_skoru']
    sayisal_tum = sayisal_cols + FE_COLS

    sayisal_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    binary_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))])
    kategorik_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    prep = ColumnTransformer([
        ('sayisal', sayisal_pipe, sayisal_tum),
        ('binary', binary_pipe, binary_cols),
        ('kategorik', kategorik_pipe, kategorik_cols),
    ], remainder='drop')

    model = Pipeline([
        ('fe', StudentFeatureEngineer()),
        ('prep', prep),
        ('model', HistGradientBoostingClassifier(random_state=42))
    ])

    model.fit(X, y)
    
    y_proba = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]

    return model, best_threshold, "HistGradientBoosting (Canlı Otomatik Eğitim)"

model, threshold, model_name = modeli_hazirla_ve_yukle()

st.sidebar.info(f"🤖 **Aktif Model:** {model_name}")

# --- KULLANICI GİRDİLERİ ---
col1, col2 = st.columns(2)

with col1:
    vize_puani = st.number_input("Vize Puanı", min_value=0.0, max_value=100.0, value=50.0)
    quiz_puani = st.number_input("Quiz Puanı", min_value=0.0, max_value=100.0, value=50.0)
    odev_puani = st.number_input("Ödev Puanı", min_value=0.0, max_value=100.0, value=60.0)
    gunluk_calisma_saati = st.slider("Günlük Çalışma Saati", min_value=0.0, max_value=12.0, value=3.0)
    gunluk_uyku_saati = st.slider("Günlük Uyku Saati", min_value=0.0, max_value=14.0, value=7.0)
    onceki_ortalama = st.number_input("Önceki Dönem Ortalaması (GPA)", min_value=0.0, max_value=4.0, value=2.5)

with col2:
    devam_yuzdesi = st.slider("Derse Devam Yüzdesi (%)", min_value=0, max_value=100, value=80)
    ders_disi_aktivite_saati = st.slider("Ders Dışı Aktivite Saati", min_value=0.0, max_value=10.0, value=1.0)
    alttan_ders_sayisi = st.number_input("Alttan Alınan Ders Sayısı", min_value=0, max_value=10, value=0)
    cinsiyet = st.selectbox("Cinsiyet", ["K", "E"])
    internet_erisimi = st.selectbox("İnternet Erişimi", ["Evet", "Hayir"])

st.markdown("---")

# --- TAHMİN ÇALIŞTIRMA ---
if st.button("🔮 Başarı Durumunu Tahmin Et", use_container_width=True):
    ogrenci_verisi = {
        'vize_puani': vize_puani,
        'quiz_puani': quiz_puani,
        'odev_puani': odev_puani,
        'gunluk_calisma_saati': gunluk_calisma_saati,
        'gunluk_uyku_saati': gunluk_uyku_saati,
        'onceki_ortalama': onceki_ortalama,
        'devam_yuzdesi': devam_yuzdesi,
        'ders_disi_aktivite_saati': ders_disi_aktivite_saati,
        'alttan_ders_sayisi': alttan_ders_sayisi,
        'cinsiyet': cinsiyet,
        'internet_erisimi': internet_erisimi
    }
    
    df_yeni = pd.DataFrame([ogrenci_verisi])
    proba = model.predict_proba(df_yeni)[:, 1][0]
    is_riskli = proba >= threshold
    
    st.subheader("📊 Tahmin Sonucu")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Kalma / Risk Olasılığı", value=f"%{proba*100:.1f}")
    with col_res2:
        st.metric(label="Optimal Karar Eşiği", value=f"{threshold:.3f}")
        
    if is_riskli:
        st.error(f"🚨 **SONUÇ: Kaldı (Riskli Öğrenci)** — Öğrencinin kalma riski yüksek (Olasılık: %{proba*100:.1f}). Akademik destek önerilir.")
    else:
        st.success(f"✅ **SONUÇ: Geçti (Başarılı)** — Öğrencinin dersi başarıyla geçmesi bekleniyor (Geçme Olasılığı: %{(1-proba)*100:.1f}).")
