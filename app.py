"""
AI Destekli CV Analiz Sitesi (Groq API ile)
--------------------------------------------
Kullanıcı PDF formatında CV'sini yükler, hedeflediği pozisyonu yazar,
Groq API (Llama modeli) ile CV analiz edilir: ATS skoru, güçlü/eksik
yönler ve geliştirme önerileri sunulur.
"""

import streamlit as st
import requests
import json
import re
from pypdf import PdfReader
import io

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
st.set_page_config(page_title="AI CV Analiz", page_icon="📄", layout="centered")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Denenecek modeller: ilki başarısız olursa (kota/hata) sıradaki denenir
MODEL_SIRASI = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
def pdf_metin_cikar(pdf_dosyasi) -> str:
    """Yüklenen PDF dosyasından metni çıkarır."""
    reader = PdfReader(io.BytesIO(pdf_dosyasi.read()))
    metin = ""
    for sayfa in reader.pages:
        metin += sayfa.extract_text() or ""
    return metin.strip()


def groq_istegi_gonder(api_key: str, prompt_text: str):
    """
    Groq API'ye istek gönderir. Sırasıyla MODEL_SIRASI listesindeki
    modelleri dener; biri hata verirse bir sonrakine geçer.
    Başarılı olursa (model_adi, yanit_metni, None) döner, olmazsa
    (None, None, hata_mesaji) döner.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    son_hata = None

    for model in MODEL_SIRASI:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.3,
        }
        try:
            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            data = response.json()

            if response.status_code == 200:
                yanit_metni = data["choices"][0]["message"]["content"]
                return model, yanit_metni, None

            hata_mesaji = data.get("error", {}).get("message", str(data))
            son_hata = f"[{model}] {hata_mesaji}"

        except Exception as e:
            son_hata = f"[{model}] Bağlantı hatası: {e}"

    return None, None, son_hata


def json_cikar(metin: str) -> dict:
    """
    Modelin döndürdüğü metinden JSON kısmını ayıklar.
    Model bazen JSON'u ```json ... ``` bloğu içine koyabiliyor, onu temizler.
    """
    temiz = re.sub(r"```json|```", "", metin).strip()
    return json.loads(temiz)


def prompt_olustur(hedef_pozisyon: str, cv_metni: str) -> str:
    return f"""
Hedef Pozisyon: {hedef_pozisyon}
CV Metni: {cv_metni}

Lütfen cevabını tam olarak şu JSON formatında ver, başka hiçbir ekstra açıklama yazma, sadece JSON döndür:
{{
    "ats_score": 85,
    "guclu_yonler": ["Maddeler halinde güçlü yönler"],
    "eksik_yonler": ["Maddeler halinde eksik veya geliştirilmesi gereken yönler"],
    "tavsiyeler": ["Eklenmesi gereken teknolojiler, projeler veya sertifika önerileri"]
}}
"""


# ---------------------------------------------------------
# ARAYÜZ
# ---------------------------------------------------------
st.title("📄 AI Destekli CV Analiz")
st.write("CV'nizi yükleyin, hedeflediğiniz rolü yazın ve yapay zeka analiz etsin!")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        help="https://console.groq.com/keys adresinden alabilirsin",
    )
    st.caption(
        "API key'ini kod içine yazmak yerine buradan gir. "
        "Böylece kodu paylaşsan bile key'in görünmez."
    )

hedef_pozisyon = st.text_input(
    "Hedeflenen İş Pozisyonu (Örn: Java Developer, Flutter Developer, IT Destek Uzmanı):"
)

yuklenen_dosya = st.file_uploader("CV'nizi PDF formatında yükleyin", type=["pdf"])

if st.button("CV'yi Analiz Et", type="primary"):
    if not api_key_input:
        st.error("Lütfen sol menüden Groq API key'ini gir.")
    elif not hedef_pozisyon:
        st.error("Lütfen hedeflenen iş pozisyonunu yaz.")
    elif not yuklenen_dosya:
        st.error("Lütfen bir PDF dosyası yükle.")
    else:
        with st.spinner("Yapay zeka CV'nizi inceliyor, lütfen bekleyin..."):
            cv_metni = pdf_metin_cikar(yuklenen_dosya)

            if not cv_metni:
                st.error(
                    "PDF'ten metin okunamadı. Dosyanın taranmış bir görüntü değil, "
                    "gerçek metin içeren bir PDF olduğundan emin ol."
                )
            else:
                prompt = prompt_olustur(hedef_pozisyon, cv_metni)
                model_adi, yanit, hata = groq_istegi_gonder(api_key_input, prompt)

                if hata and not yanit:
                    st.error(f"Groq API Hatası: {hata}")
                    st.info(
                        "💡 API key'inin doğru kopyalandığından ve "
                        "https://console.groq.com/keys üzerinden aktif olduğundan emin ol."
                    )
                else:
                    try:
                        sonuc = json_cikar(yanit)

                        st.success(f"Analiz tamamlandı! (Kullanılan model: {model_adi})")

                        st.subheader("🎯 ATS Uyum Skoru")
                        st.progress(sonuc["ats_score"] / 100)
                        st.metric("Skor", f"{sonuc['ats_score']} / 100")

                        st.subheader("✅ Güçlü Yönler")
                        for madde in sonuc["guclu_yonler"]:
                            st.markdown(f"- {madde}")

                        st.subheader("⚠️ Eksik / Geliştirilmesi Gereken Yönler")
                        for madde in sonuc["eksik_yonler"]:
                            st.markdown(f"- {madde}")

                        st.subheader("💡 Tavsiyeler")
                        for madde in sonuc["tavsiyeler"]:
                            st.markdown(f"- {madde}")

                    except (json.JSONDecodeError, KeyError) as e:
                        st.error(f"Yapay zekadan gelen cevap işlenemedi: {e}")
                        st.text_area("Ham cevap:", yanit, height=200)