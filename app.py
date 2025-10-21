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

# ... onceki kodlar (load_dotenv, load_dataset, data_df, print(data_df.head())) ...

# --- 2. VERI SETI HAZIRLAMA (Devam: RAG Icin Hazirlik) ---

from langchain_community.document_loaders.dataframe import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Yerel bir model kullanimi
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# ----------------------------------------------------------------------
# 2.1. DataFrame'i LangChain Belgelerine (Document) Dönüştürme
# RAG sisteminde kaynak olarak 'yorum' kolonunu kullanacagiz.

# Her satir icin bir LangChain Document objesi olusturulur.
# 'page_content' = Yorum metni (kaynak bilgi)
# 'metadata' = kitap_adi, yazar ve rating gibi ek bilgiler

documents = []
for index, row in data_df.iterrows():
    # Sadece metin iceren ve kayitli yorumlari aliyoruz (NaN kontrolu)
    if pd.notna(row['yorum']):
        doc = Document(
            page_content=row['yorum'],
            metadata={
                "kitap_adi": row['kitap_adi'],
                "yazar": row['yazar'],
                "rating": row['rating']
            }
        )
        documents.append(doc)

print(f"\nToplam olusturulan Document sayisi: {len(documents)}")

# ----------------------------------------------------------------------
# 2.2. Parçalama (Chunking)
# Uzun yorumlari kucuk ve yonetilebilir parcalara ayirarak baglamsal dogrulugu artirma.

# recursive splitter, metni belirtilen ayiricilarla (varsayilan: \n\n, \n, " ", "") parcalar.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Her bir parcada maksimum 1000 karakter
    chunk_overlap=200, # Parcalarin %20'si (200 karakter) birbiriyle cakissin (baglami korumak icin)
    length_function=len,
)

chunks = text_splitter.split_documents(documents)

print(f"Toplam yorumdan olusan parca (chunk) sayisi: {len(chunks)}")

# ----------------------------------------------------------------------
# 2.3. Gömme (Embedding) ve Vektör Veritabanı Oluşturma
# Metin parcalarini, LLM'in anlayacagi sayisal vektorlere donusturme.

# **EMBEDDING MODELI SECIMI:**
# HuggingFace (Yerel) Embedding Modelini seciyoruz. (Hata riskini azaltir, API gerektirmez)
# Yüksek performans icin 'all-MiniLM-L6-v2' yaygin ve hizli bir secimdir.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vektör veritabanini (ChromaDB) olusturma ve parcalari indeksleme
# Bu islem uzun surebilir. 'persist_directory' ile DB'yi kaydederiz.
persist_directory = './chroma_db'
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Indekslemeyi tamamla ve DB'yi diske kaydet
vector_store.persist()
print(f"Chroma Vektör Veritabanı oluşturuldu ve {persist_directory} klasörüne kaydedildi.")

# --- 4. COZUM MIMARISI (Hazirlik) ---
# Bu asama RAG mimarisinin 'Retrieval' (Geri Getirme) adimini olusturur.
# Sırada 'Generation' (Uretme) adimi ve LLM entegrasyonu var.
# ...
# ... onceki kodlar (ChromaDB olusturma ve persist kodu) ...

# --- 4. ÇÖZÜM MIMARINIZ (RAG Zinciri) ---

# Gerekli LangChain ve Gemini kütüphanelerini içe aktar
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# ----------------------------------------------------------------------
# 4.1. LLM ve Prompt Hazırlığı

# API anahtarini ortam degiskeninden yukle
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("HATA: GEMINI_API_KEY, .env dosyasindan yuklenemedi veya tanimli degil.")
    # API anahtari olmadan LLM calismaz.

# LLM'i tanimla (Gemini 2.0 Flash)
# 'google_api_key' parametresi ile anahtari dogrudan iletmek hata riskini azaltir.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1 # Daha tutarli ve gercekci cevaplar icin dusuk sicaklik
)

# RAG icin Prompt Template (Modelin cevabini yonlendirmek icin)
prompt_template = """
Sen, kitap yorumlarindan bilgi cikarmak uzere tasarlanmis bir uzmansin.
Yalnizca asagidaki 'Baglam' icinde verilen bilgilere dayanarak kullanicinin sorusunu cevapla. 
Eger baglamda yeterli bilgi yoksa, "Uzgunum, bu soruyla ilgili yeterli yorum bilgisine sahip degilim." diye cevap ver.
Cevaplarini akici ve dogal bir dilde olustur.

Baglam:
{context}

Soru: {question}
"""
PROMPT = ChatPromptTemplate.from_template(prompt_template)

# ----------------------------------------------------------------------
# 4.2. RAG Zincirini (Chain) Kurma

# Vektor veritabanini (vector_store) bir Retriever (Geri Getirici) olarak kullan
# Bu, kullanici sorgusuna en yakin belgeleri sececek.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3} # Sorguya en yakin 3 belgeyi getir
)

# RetrievalQA zinciri, RAG akisini yonetir:
# 1. Kullanici sorusu gelir.
# 2. Retriever (vector_store) ilgili 'context'u bulur.
# 3. Prompt, 'context' ve 'question' ile birlestirilir.
# 4. LLM (Gemini) cevabi uretir.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Tum kaynaklari tek bir prompt'a doldurur
    retriever=retriever,
    return_source_documents=True, # Kaynak belgeleri (yorum metinleri) dondurur
    chain_type_kwargs={"prompt": PROMPT}
)

# ----------------------------------------------------------------------
# 4.3. Test Sorgusu

print("\n--- RAG Sistemi Baslatiliyor (Test Sorgusu) ---")
test_query = "En cok hangi kitaplar hakkında olumlu yorumlar var ve bu yorumlardan biri nedir?"

result = qa_chain.invoke(test_query)

print(f"\nSoru: {test_query}")
print(f"\nCevap (Gemini): {result['result']}")
print("\n--- Kaynak Belgeler (Source Documents) ---")

# Kaynak belgelerin metadata'larini (kitap adı, yazar) goruntule
for doc in result['source_documents']:
    print(f"Kitap Adı: {doc.metadata.get('kitap_adi', 'Bilinmiyor')}")
    print(f"Yazar: {doc.metadata.get('yazar', 'Bilinmiyor')}")
    print(f"Rating: {doc.metadata.get('rating', 'Bilinmiyor')}")
    print("-" * 20)

# --- 5. WEB ARAYÜZÜ (Sirada) ---
# Sirada bu 'qa_chain'i Streamlit/Gradio ile bir web arayuzune tasimak var.
