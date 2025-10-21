# --- 1. GEREKLILIKLERI VE KUTUPHANELERI ICERI AKTARMA ---
# Tum teknik anlatimlar burada yorum satirlari icinde verilmektedir. [cite: 7]
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
import os
import streamlit as st # Streamlit arayuzu icin

# LangChain Kütüphaneleri
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval_qa import RetrievalQA

# --- GLOBAL AYARLAR ---
# Veritabaninin kaydedilecegi dizin
PERSIST_DIRECTORY = './chroma_db'
# Embedding modeli
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# LLM modeli
LLM_MODEL_NAME = "gemini-2.0-flash"

# --- 2. VERI SETI VE RAG BILESENLERINI YUKLEME/OLUSTURMA ---

@st.cache_resource # Streamlit'in bileşenleri yeniden yüklemesini önler
def initialize_rag_components():
    """
    RAG sisteminin kritik bilesenlerini (Veri Seti, Vektor DB, LLM) hazirlar.
    """
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY, .env dosyasindan yuklenemedi. Lutfen kontrol edin.")
        return None

    # Vektor DB'nin varligini kontrol etme
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        st.info("Vektör Veritabanı (ChromaDB) diske kaydedilmis haliyle yukleniyor...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
    else:
        st.warning("Vektör Veritabanı diskte bulunamadi, yeniden olusturuluyor (Bu islem zaman alabilir)...")
        
        # Veri setini yukle (Token ile)
        try:
            dataset = load_dataset("alibayram/kitapyurdu_yorumlar", token=HF_TOKEN)
            data_df = pd.DataFrame(dataset['train'])
        except Exception as e:
            st.error(f"Veri seti yuklenirken hata olustu. Token'inizi kontrol edin: {e}")
            return None

        # LangChain Document'lara donusturme
        documents = []
        for index, row in data_df.iterrows():
            if pd.notna(row['yorum']):
                doc = Document(page_content=row['yorum'], metadata={"kitap_adi": row['kitap_adi'], "yazar": row['yazar'], "rating": row['rating']})
                documents.append(doc)

        # Parçalama (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_documents(documents)
        
        # Gömme (Embedding) ve Vektör DB olusturma
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
        vector_store.persist()
        st.success("Vektör Veritabanı basariyla olusturuldu ve diske kaydedildi.")

    # LLM tanimlama (Gemini 2.0 Flash)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0.1)

    # Prompt Template (Daha onceki gibi)
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

    # RAG Zincirini kurma
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
    
    return qa_chain

# --- 3. STREAMLIT ARAYUZU VE CALISMA AKISI ---

st.title("📚 Kitapyurdu Yorum Analizi Chatbot (RAG)") [cite: 24, 25]
st.write("Bu chatbot, Kitapyurdu'ndaki binlerce yorumu analiz ederek sorularinizi Gemini 2.0 Flash ile cevaplar.")

# RAG bilesenlerini yukle (Sadece bir kez calisir)
qa_chain = initialize_rag_components()

if qa_chain:
    # Kullanici girdisi
    user_query = st.text_input("Sorgunuzu Girin:", placeholder="Örneğin: En çok hangi kitaplar önerilmiş?")

    # Sorgulama dugmesi
    if st.button("Sorgula"):
        if user_query:
            with st.spinner("Cevap araniyor ve uretiliyor..."):
                try:
                    # RAG zincirini calistir
                    result = qa_chain.invoke(user_query)

                    # Cevap alanlari
                    st.header("Cevap")
                    st.success(result['result'])

                    # Kaynak belgeleri goster
                    st.header("Kullanilan Kaynak Yorumlar")
                    for i, doc in enumerate(result['source_documents']):
                        with st.expander(f"Kaynak {i+1}: {doc.metadata.get('kitap_adi', 'Bilinmiyor')} - Yazar: {doc.metadata.get('yazar', 'Bilinmiyor')}"):
                            st.write(f"**Derecelendirme (Rating):** {doc.metadata.get('rating', 'Bilinmiyor')}")
                            st.markdown(f"**Yorum Metni:** {doc.page_content}")
                            
                except Exception as e:
                    st.error(f"Sorgu islenirken bir hata olustu: {e}")
                    st.warning("API anahtarinizin dogru oldugundan ve model erisiminizin bulundugundan emin olun.")
        else:
            st.warning("Lutfen bir sorgu girin.")
