import chromadb
import os
import time
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# === KONFIGURACJA ŚCIEZEK ===
DB_DIR = "../.././../wektorowa_bazadanych/lokalna"
#collection_name = "documents"
#DOKUMENTY_PATH = "../dokumenty"

def init_chroma():
    os.makedirs(DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=DB_DIR)

def check_collection_exists(client, collection_name):
    """Sprawdza czy kolekcja istnieje i zawiera dane"""
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        if count > 0:
            print(f"✓ Kolekcja '{collection_name}' istnieje i zawiera {count} dokumentów")
            return True
        else:
            print(f"✗ Kolekcja '{collection_name}' istnieje ale jest pusta")
            return False
    except Exception as e:
        print(f"✗ Kolekcja '{collection_name}' nie istnieje")
        return False

def save_to_vectorstore(documents, embeddings, client, collection_name):
    """Zapis dokumentów do bazy wektorowej"""
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=client,
        collection_name=collection_name,
    )
    
def load_vectorstore(client, collection_name, embeddings):
    print("--- Ładowanie bazy wektorowej ---")
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )

# def load_pdf_documents(documents_path):
#     """Ładowanie dokumentów PDF z folderu"""
#     print("--- Ładowanie PDFów ---")
#     docs = []
#     if not os.path.exists(documents_path):
#         return []
#     for file in os.listdir(documents_path):
#         if file.endswith(".pdf"):
#             try:
#                 loader = PyPDFLoader(os.path.join(documents_path, file))
#                 docs.extend(loader.load())
#                 print(f"✓ Załadowano: {file}")
#             except Exception as e:
#                 print(f"✗ Błąd z {file}: {e}")
#     return docs

def load_pdf_documents(documents_path):
    """Ładowanie dokumentów PDF z folderu"""
    print("--- Ładowanie PDFów ---")
    docs = []
    if not os.path.exists(documents_path):
        return []
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(documents_path, file),"utf-8")
                docs.extend(loader.load())
                print(f"✓ Załadowano: {file}")
            except Exception as e:
                print(f"✗ Błąd z {file}: {e}")
    return docs



def index_documents_in_batches(splits, embeddings, client, collection_name, batch_size=50):
    """Indeksowanie z postępem"""
    print("--- Indeksowanie w ChromaDB ---")
    start_time = time.time()
    vectorstore = None
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        vectorstore = save_to_vectorstore(batch, embeddings, client, collection_name)
        print(f"Zaindeksowano {min(i + batch_size, len(splits))}/{len(splits)} fragmentów")
    
    print(f"⏱ Czas indeksowania: {time.time() - start_time:.2f}s")
    return vectorstore
