# --- 2. VERI SETI HAZIRLAMA (Guncelleme: Token Gereksinimi) ---

# Veri setine erisim icin Hugging Face token kullanilmaktadir.
# Token, '.env' dosyasindan yuklenerek guvenli bir sekilde kullanilmaktadir.

from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv # Yeni kütüphane
import os

# Ortam degiskenlerini yukle (.env dosyasindan)
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Eğer token yuklenemediyse uyari ver
if not HF_TOKEN:
    print("HATA: HUGGINGFACE_TOKEN, .env dosyasindan yuklenemedi veya tanimli degil.")
    # Token olmadan devam edemeyecegimiz icin programi durdurabiliriz.
    # exit() 

# Veri setini token kullanarak yukle
try:
    # Veri setini yuklerken 'token' parametresini kullanmak, kimlik dogrulamayi saglar.
    dataset = load_dataset("alibayram/kitapyurdu_yorumlar", token=HF_TOKEN)
    data_df = pd.DataFrame(dataset['train'])

    print("Veri seti basariyla yuklendi. Ilk 5 satir:")
    print(data_df.head())

except Exception as e:
    print(f"Veri seti yuklenirken hata olustu: {e}")
    # Token hatasi veya baglanti hatasi olabilir.

# --- ... RAG pipeline kodlari devam edecek.
