from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Öğrenci Başarı ve Risk Tahmin Sistemi", page_icon="🎓", layout="centered")

# pkl dosyası app.py ile AYNI klasörde (repo kökü) olmalı
MODEL_YOLU = Path(__file__).parent / "ogrenci_basari_production_model.pkl"


# --- Özel sınıf: pkl bu sınıfı '__main__.StudentFeatureEngineer' olarak arıyor ---
# joblib.load'dan ÖNCE tanımlanmalı. Colab'daki sınıfla birebir aynı olmalı.
class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        d['ortalama_akademik_puan'] = d[['vize_puani', 'quiz_puani', 'odev_puani']].mean(axis=1, skipna=True)
        d['calisma_uyku_orani'] = d['gunluk_calisma_saati'] / (d['gunluk_uyku_saati'] + 1)
        d['basari_devam_skoru'] = d['onceki_ortalama'] * (d['devam_yuzdesi'] / 100)
        d['toplam_zaman_yuku'] = d[['gunluk_calisma_saati', 'ders_disi_aktivite_saati']].sum(axis=1, skipna=True)
        d['akademik_risk_skoru'] = d['alttan_ders_sayisi'] / d['onceki_ortalama'].replace(0, 0.1)
        return d.replace([np.inf, -np.inf], np.nan)


@st.cache_resource
def model_yukle():
    paket = joblib.load(MODEL_YOLU)
    return paket['model'], paket['threshold'], paket.get('model_name', 'Model'), paket['feature_names']


if not MODEL_YOLU.exists():
    st.error(f"Model dosyası bulunamadı: {MODEL_YOLU.name}")
    st.write("Bu klasördeki dosyalar:", sorted(p.name for p in MODEL_YOLU.parent.iterdir()))
    st.stop()

try:
    model, threshold, model_name, kolonlar = model_yukle()
except Exception as e:
    st.error(f"⚠️ Model yüklenirken hata oluştu: {type(e).__name__}: {e}")
    st.info("Çoğunlukla scikit-learn sürüm uyuşmazlığıdır: requirements.txt içinde scikit-learn==1.6.1 olmalı.")
    st.stop()

st.title("🎓 Öğrenci Başarı ve Risk Tahmin Asistanı")
st.caption(f"Aktif model: {model_name}. Bu bir karar destek aracıdır, kesin hüküm değildir.")
st.markdown("---")

# Kategori seçenekleri eğitimdeki değerlerle BİREBİR aynı olmalı (Hayır, Diğer vb.)
col1, col2 = st.columns(2)
with col1:
    yas = st.number_input("Yaş", min_value=15, max_value=60, value=20, step=1)
    cinsiyet = st.selectbox("Cinsiyet", ["Kadın", "Erkek", "Diğer"])
    vize_puani = st.number_input("Vize Puanı", min_value=0.0, max_value=100.0, value=50.0)
    quiz_puani = st.number_input("Quiz Puanı", min_value=0.0, max_value=100.0, value=50.0)
    odev_puani = st.number_input("Ödev Puanı", min_value=0.0, max_value=100.0, value=60.0)
    onceki_ortalama = st.number_input("Önceki Dönem Ortalaması", min_value=0.0, max_value=4.0, value=2.5)
    alttan_ders_sayisi = st.number_input("Alttan Alınan Ders Sayısı", min_value=0, max_value=10, value=0, step=1)

with col2:
    gunluk_calisma_saati = st.slider("Günlük Çalışma Saati", 0.0, 12.0, 3.0, 0.5)
    gunluk_uyku_saati = st.slider("Günlük Uyku Saati", 0.0, 14.0, 7.0, 0.5)
    devam_yuzdesi = st.slider("Derse Devam Yüzdesi (%)", 0, 100, 80)
    ders_disi_aktivite_saati = st.slider("Ders Dışı Aktivite Saati", 0.0, 10.0, 1.0, 0.5)
    internet_erisimi = st.selectbox("İnternet Erişimi", ["Evet", "Hayır"])
    ebeveyn_egitim_durumu = st.selectbox(
        "Ebeveyn Eğitim Durumu", ["Lise", "Diploma", "Lisans", "Yüksek Lisans", "Doktora"])

st.markdown("---")

if st.button("🔮 Başarı Durumunu Tahmin Et", use_container_width=True):
    girdi = {
        'yas': yas, 'cinsiyet': cinsiyet,
        'gunluk_calisma_saati': gunluk_calisma_saati, 'devam_yuzdesi': devam_yuzdesi,
        'gunluk_uyku_saati': gunluk_uyku_saati, 'odev_puani': odev_puani,
        'quiz_puani': quiz_puani, 'vize_puani': vize_puani,
        'onceki_ortalama': onceki_ortalama, 'alttan_ders_sayisi': alttan_ders_sayisi,
        'internet_erisimi': internet_erisimi,
        'ders_disi_aktivite_saati': ders_disi_aktivite_saati,
        'ebeveyn_egitim_durumu': ebeveyn_egitim_durumu,
    }
    df_yeni = pd.DataFrame([girdi])[kolonlar]  # sütun sırası eğitimdekiyle aynı
    proba = float(model.predict_proba(df_yeni)[:, 1][0])

    st.subheader("📊 Tahmin Sonucu")
    c1, c2 = st.columns(2)
    c1.metric("Kalma risk skoru", f"%{proba * 100:.1f}")
    c2.metric("Karar eşiği", f"%{threshold * 100:.1f}")
    st.progress(min(max(proba, 0.0), 1.0))

    if proba >= threshold:
        st.error("🚨 **Kaldı (Riskli Öğrenci)**: kalma riski yüksek. Akademik destek önerilir.")
    else:
        st.success("✅ **Geçti (Başarılı)**: dersi geçmesi bekleniyor.")
