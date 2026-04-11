import json
import os
import time
import psutil  # Biblioteka do monitorowania zasobów
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from tqdm import tqdm


# === KONFIGURACJA ===
MODEL_DIR = "model_llm"
CONFIG_FILE = "modele.json"
N_THREADS = 4


def load_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_memory_usage():
    """Zwraca aktualne użycie RAM przez proces w MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(question, vectorstore):
    config = load_config()
    model_keys = list(config["models"].keys())

    print(f"🚀 Rozpoczynam benchmark dla {len(model_keys)} modeli.")

    for model_key in model_keys:
        print(f"\n{'=' * 20}")
        print(f"🤖 MODEL: {model_key}")
        print(f"{'=' * 20}")

        # 1. Nadpisujemy tymczasowo wybrany model w konfiguracji
        # (Można też zmodyfikować get_llm, by przyjmował model_key jako argument)
        current_llm = get_llm(model_key, config)

        # 2. Wykonujemy zapytanie RAG
        # Przekazujemy model_key bezpośrednio, aby ask_rag wiedział co loguje
        answer = ask_rag(question, vectorstore, current_llm, model_key, config)

        print(f"Odp: {answer}")

        # 3. CZYSZCZENIE PAMIĘCI
        # To kluczowe, by zwolnić RAM przed załadowaniem kolejnego modelu
        del current_llm
        import gc
        gc.collect()
        time.sleep(2)  # Krótka pauza na "oddech" systemu

    print("\n✅ Wszystkie modele zostały przetestowane.")


# Pomocnicze funkcje dostosowane do pętli:

def get_llm(model_key, config):
    model_cfg = config["models"][model_key]
    repo_id = model_cfg["repo_id"]
    filename = model_cfg["filename"]

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(model_path):
        print(f"Pobieranie {filename}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODEL_DIR)

    mem_before = get_memory_usage()
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=N_THREADS,
        n_gpu_layers=0,
        verbose=False
    )
    mem_after = get_memory_usage()
    print(f"✓ Załadowano. Przyrost RAM: {mem_after - mem_before:.2f} MB")
    return llm


def ask_rag(question, vectorstore, llm, model_key, config):
    start_time = time.time()
    template = config["models"][model_key]["prompt_template"]

    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs]) if docs else "Brak kontekstu."
    prompt = template.format(context=context, question=question)

    response = llm(prompt, max_tokens=200, temperature=0.0)
    answer = response['choices'][0]['text'].strip()

    elapsed = time.time() - start_time
    mem_end = get_memory_usage()

    save_result_to_json(model_key, question, answer, elapsed, mem_end)
    return answer


def save_result_to_json(model_name, question, answer, duration, ram_usage):
    """Logowanie odpowiedzi wraz z wydajnością"""
    res_file = "results.json"
    results = {}
    if os.path.exists(res_file):
        try:
            with open(res_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            results = {}

    entry_id = f"{model_name}_{int(time.time())}"
    results[entry_id] = {
        "model": model_name,
        "question": question,
        "answer": answer,
        "metrics": {
            "duration_sec": round(duration, 2),
            "ram_usage_mb": round(ram_usage, 2)
        }
    }

    with open(res_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)