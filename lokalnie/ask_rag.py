from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
# === KONFIGURACJA MODELU LLM ===
MODEL_DIR = "model_llm"
MODEL_REPO = "bartowski/gemma-2-9b-it-GGUF"
MODEL_FILE = "gemma-2-9b-it-IQ2_M.gguf"

# !pip install llama-cpp-python


llm = Llama.from_pretrained(
	repo_id=MODEL_REPO,
	filename=MODEL_FILE,
)

N_THREADS = 4  # Dla Ryzen 5 3500U


def get_llm():
    """Pobieranie i inicjalizacja modelu LLM"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)

    if not os.path.exists(model_path):
        print(f"Pobieranie modelu {MODEL_FILE} ...")
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=MODEL_DIR
        )
        print(f"✓ Model pobrany do: {downloaded_path}")
    else:
        print("✓ Model już jest w pamięci lokalnej")

    print("--- Ładowanie modelu  ---")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Zoptymalizowane dla CPU
            n_threads=N_THREADS,
            n_batch=128,
            n_gpu_layers=0,  # CPU only
            use_mmap=True,  # Lepsza wydajność
            use_mlock=False,
            verbose=False
        )
        print("✓ Model załadowany pomyślnie")
        return llm
    except Exception as e:
        print(f"✗ Błąd ładowania modelu: {e}")
        raise e


def ask_rag(question, vectorstore, llm):
    """Zoptymalizowana funkcja RAG z ograniczeniem czasowym"""
    import time
    start_time = time.time()

    try:
        # Wyszukiwanie z ograniczeniem do 3 wyników dla szybkości
        docs = vectorstore.similarity_search(question, k=3)

        if not docs:
            return "Przepraszam, nie znalazłem odpowiedzi w dostępnych dokumentach."

        context = "\n".join([d.page_content for d in docs])

        # Prompt pod Gemma-2-9b-it-GGUF
        prompt = f"""<start_of_turn>user
        Jesteś asystentem akademickim. Odpowiedz na pytanie krótko i konkretnie, korzystając WYŁĄCZNIE z podanego kontekstu. Odpowiadaj w języku polskim.

        KONTEKST:
        {context}

        PYTANIE:
        {question}<end_of_turn>
        <start_of_turn>model
        """

        response = llm(
            prompt,
            max_tokens=150,
            stop=["<|im_end|>"],
            echo=False,
            temperature=0.0,
            top_p=0.9
        )

        elapsed = time.time() - start_time
        answer = response['choices'][0]['text'].strip()

        print(f"⏱ Czas generowania: {elapsed:.2f}s")
        return answer

    except Exception as e:
        return f"Przepraszam, wystąpił błąd podczas przetwarzania: {str(e)}"
