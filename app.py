import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. ÖZEL FEATURE ENGINEER SINIFI (Modelin çalışması için zorunludur) ---
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

# --- 2. STREAMLIT SAYFA YAPISI VE STİLİ ---
st.set_page_config(
    page_title="Öğrenci Başarı ve Risk Tahmin Sistemi",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Öğrenci Başarı ve Risk Tahmin Paneli")
st.markdown("Bu uygulama, öğrenci verilerini analiz ederek **Geçme / Kalma** durumunu yapay zeka (Soft Voting) modeliyle tahmin eder.")

@st.cache_resource
def model_yukle():
    # Kaydettiğimiz model dosyasının adı
    paket = joblib.load('ogrenci_basari_production_model.pkl')
    return paket['model'], paket['threshold']

try:
    model, threshold = model_yukle()
except Exception as e:
    st.error(f"Model dosyası ('ogrenci_basari_production_model.pkl') yüklenemedi! Lütfen model dosyasını GitHub reposuna eklediğinizden emin olun. Hata: {e}")
    st.stop()

# --- 3. KULLANICI GİRDİLERİ (SIDEBAR & FORM) ---
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

# Modelin beklediği temel sözlük yapısı
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

# --- 4. TAHMİN ÇALIŞTIRMA ---
if st.button("🚀 Başarı Durumunu Tahmin Et", type="primary"):
    df_input = pd.DataFrame([giris_verisi])
    
    try:
        # Olasılık tahmini (1: Kaldı sınıfı)
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
