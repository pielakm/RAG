from langchain_huggingface import HuggingFaceEmbeddings

# === KONFIGURACJA EMBEDDING ===
MODEL_SAVE_PATH = "model_embedding"
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True, 'batch_size': 8}

def get_embeddings():
    print("--- Inicjalizacja modelu embeddingowego ---")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=MODEL_SAVE_PATH,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        multi_process=False  # Wyłącz multiprocessing dla CPU
    )
    
