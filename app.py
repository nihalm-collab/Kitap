# PROJE AMACI: Kitapyurdu yorumlarini FAISS ve Gemini ile sorgulayan RAG Chatbot.

import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset

# Yeni Kutuphaneler
from google import genai
from google.genai.errors import APIError
import faiss # Vektor veritabani
import numpy as np
import tiktoken # Parcalama icin

# --- GLOBAL AYARLAR ---
EMBEDDING_MODEL = 'text-embedding-004'
LLM_MODEL = "gemini-2.0-flash"
DB_FILE = "faiss_index.bin"

# --- 2. VERI SETI VE RAG BILESENLERINI YUKLEME/OLUSTURMA ---

@st.cache_resource
def initialize_rag_components():
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY, .env dosyasindan yuklenemedi veya tanimli degil.")
        return None

    try:
        # Gemini API İstemcisini Başlatma
        client = genai.Client(api_key=GEMINI_API_KEY)
        st.info(f"LLM ve Embedding Modeli: {LLM_MODEL} ve {EMBEDDING_MODEL}")
        
        # Eğer FAISS indeksi diskte varsa yükle
        if os.path.exists(DB_FILE):
            st.info("FAISS İndeksi diske kaydedilmis haliyle yukleniyor...")
            index = faiss.read_index(DB_FILE)
            with open("metadata.txt", "r") as f:
                documents = f.read().split("\n---\n") # Metin parcalari
        else:
            st.warning("FAISS İndeksi diskte bulunamadi, yeniden olusturuluyor (Bu islem zaman alabilir)...")

            # Veri Setini Yükle
            dataset = load_dataset("alibayram/kitapyurdu_yorumlar", token=HF_TOKEN)
            data_df = pd.DataFrame(dataset['train'])

            # ----------------------------------------------------------------------
            # Parçalama (Chunking) - tiktoken kullanarak
            tokenizer = tiktoken.get_encoding("cl100k_base")
            max_chunk_size = 500
            
            documents = []
            metadata_list = []
            
            for index, row in data_df.iterrows():
                if pd.notna(row['yorum']):
                    text = row['yorum']
                    tokens = tokenizer.encode(text)
                    
                    # Basit Parcalama
                    for i in range(0, len(tokens), max_chunk_size):
                        chunk_tokens = tokens[i:i + max_chunk_size]
                        chunk_text = tokenizer.decode(chunk_tokens)
                        
                        documents.append(chunk_text)
                        metadata_list.append({
                            "kitap_adi": row['kitap_adi'], 
                            "yazar": row['yazar'], 
                            "rating": row['rating']
                        })

            # ----------------------------------------------------------------------
            # Gömme (Embedding) ve FAISS Oluşturma
            
            # 1. Metinleri Vektörlere Dönüştürme
            # Batch ile gonderme hizli olabilir. Teker teker gonderiyoruz.
            embeddings_list = []
            for doc in documents:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    content=doc
                )
                embeddings_list.append(response['embedding'])
                
            embeddings_np = np.array(embeddings_list).astype('float32')

            # 2. FAISS İndeksi Oluşturma
            dimension = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_np)
            
            # 3. İndeksi diske kaydetme
            faiss.write_index(index, DB_FILE)
            with open("metadata.txt", "w") as f:
                f.write("\n---\n".join(documents)) # Metinleri de kaydet
            
            st.success(f"FAISS İndeksi başarıyla oluşturuldu ve {DB_FILE} dosyasına kaydedildi. Toplam parça: {len(documents)}")

        return client, index, documents

    except APIError as e:
        st.error(f"Gemini API Hatası: Lütfen GEMINI_API_KEY'inizi kontrol edin. {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Beklenmedik bir hata oluştu: {e}")
        return None, None, None

# --- 3. STREAMLIT ARAYUZU VE CALISMA AKISI ---

st.title("📚 Kitapyurdu Yorum Analizi Chatbot (FAISS RAG)")
st.write("Bu chatbot, Kitapyurdu yorumlarini FAISS ile arar ve sorularinizi Gemini 2.0 Flash ile cevaplar.")

# RAG bilesenlerini yukle (Sadece bir kez calisir)
rag_components = initialize_rag_components()

if rag_components and rag_components[0] is not None:
    client, index, documents = rag_components
    
    # Prompt Template (Daha onceki gibi)
    PROMPT_TEMPLATE = """
    Sen, kitap yorumlarindan bilgi cikarmak uzere tasarlanmis bir uzmansin.
    Yalnizca asagidaki 'Baglam' icinde verilen bilgilere dayanarak kullanicinin sorusunu cevapla. 
    Eger baglamda yeterli bilgi yoksa, "Uzgunum, bu soruyla ilgili yeterli yorum bilgisine sahip degilim." diye cevap ver.
    Cevaplarini akici ve dogal bir dilde olustur.

    Baglam:
    {context}

    Soru: {question}
    """

    user_query = st.text_input("Sorgunuzu Girin:", placeholder="Örneğin: En çok hangi kitaplar önerilmiş?")

    if st.button("Sorgula"):
        if user_query:
            with st.spinner("Cevap araniyor ve uretiliyor..."):
                try:
                    # 1. Sorguyu Gömme (Embed the Query)
                    query_embedding = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        content=user_query
                    )['embedding']
                    query_vector = np.array([query_embedding]).astype('float32')

                    # 2. FAISS Arama (Retrieval)
                    k = 3 # En yakin 3 parcayi getir
                    distances, indices = index.search(query_vector, k)
                    
                    # 3. Context Olusturma
                    source_chunks = [documents[i] for i in indices[0]]
                    context = "\n---\n".join(source_chunks)

                    # 4. Prompt ve Generation
                    formatted_prompt = PROMPT_TEMPLATE.format(context=context, question=user_query)
                    
                    response = client.models.generate_content(
                        model=LLM_MODEL,
                        contents=[formatted_prompt]
                    )

                    # Cevap alanlari
                    st.header("Cevap")
                    st.success(response.text)

                    # Kaynak belgeleri goster
                    st.header("Kullanilan Kaynak Yorum Parçaları")
                    for i, chunk in enumerate(source_chunks):
                        with st.expander(f"Kaynak Parça {i+1}"):
                            st.markdown(chunk)
                            
                except APIError as e:
                    st.error(f"Gemini API Hatası: {e}")
                except Exception as e:
                    st.error(f"Sorgu işlenirken bir hata oluştu: {e}")
        else:
            st.warning("Lutfen bir sorgu girin.")
