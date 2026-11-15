import streamlit as st
import pandas as pd
import joblib

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Öğrenci Başarı ve Risk Tahmin Sistemi",
    page_icon="🎓",
    layout="centered"
)

# Modeli, Eşik Değerini ve Model Adını Yükleme
@st.cache_resource
def model_yukle():
    paket = joblib.load('ogrenci_basari_production_model(2).pkl')
    return paket['model'], paket['threshold'], paket.get('model_name', 'Optimizasyon Modeli')

try:
    model, threshold, model_name = model_yukle()
except Exception as e:
    st.error(f"⚠️ Model dosyası ('ogrenci_basari_production_model.pkl') bulunamadı! Lütfen önce model eğitim kodunu çalıştırıp dosyayı dizine ekleyin.")
    st.stop()

st.title("🎓 Öğrenci Başarı ve Risk Tahmin Asistanı")
st.markdown(f"**Aktif Yapay Zeka Modeli:** `{model_name}` (En yüksek F1 skoruna göre otomatik seçilmiştir)")
st.markdown("Aşağıdaki akademik ve kişisel bilgileri girerek öğrencinin **Geçme / Kalma** durumunu anlık tahmin edin.")

st.markdown("---")

# Kullanıcıdan Girdi Alma Alanları
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

# Tahmin Butonu
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
    
    # Olasılık Hesaplama
    proba = model.predict_proba(df_yeni)[:, 1][0]
    is_riskli = proba >= threshold
    
    # Sonuç Gösterimi
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
