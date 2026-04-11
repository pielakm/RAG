# 📚 Baza Wektorowa - Lokalny System RAG

Projekt implementujący **Retrieval Augmented Generation (RAG)** z lokalną bazą wektorową, embeddingami oraz modelami LLM działającymi w całości offline.

---

## 🎯 Opis Projektu

System umożliwia zadawanie pytań dotyczących dokumentów akademicznych (PDF, TXT) z odpowiedziami generowanymi przez lokalne modele sztucznej inteligencji. Projekt zawiera:

✅ **Baza wektorowa** (ChromaDB) - przechowywanie i wyszukiwanie dokumentów  
✅ **Embeddingi** (sentence-transformers) - reprezentacja semantyczna tekstu  
✅ **Modele LLM** (GGUF format) - generowanie odpowiedzi  
✅ **Benchmarking** - porównanie wydajności różnych modeli  
✅ **Przetwarzanie dokumentów** - PDFy i pliki tekstowe  

---

## 📁 Struktura Projektu

```
baza_wektorowa/
├── lokalnie/                    # Główny kod aplikacji
│   ├── main_lokalnie.py        # Punkt wejścia
│   ├── database.py             # Operacje na ChromaDB
│   ├── embedding.py            # Model embeddingowy
│   ├── ask_rag.py             # Logika RAG
│   ├── ask_rag__from_json_.py # Benchmark modeli z JSON
│   ├── chunks.py              # Chunkowanie dokumentów
│   ├── advanced_rag.py        # Zaawansowane funkcje RAG
│   ├── modele.json            # Konfiguracja modeli LLM
│   ├── dostrajanie/           # Moduł do fine-tuningu
│   └── __pycache__/           # Cache Pythona
│
├── RAG_TEST/                   # Testy i eksperymenty
│   ├── RAG.ipynb              # Notebook z testami
│   ├── bazawektorowa_pdf/     # ChromaDB - baza wektorowa
│   └── modele.json            # Testowa konfiguracja
│
├── dokumenty/                  # Dokumenty źródłowe
│   ├── pdfy/                  # Pliki PDF
│   │   ├── 1.pdf
│   │   ├── 2.pdf
│   │   └── 3.pdf
│   └── txt/                   # Pliki tekstowe
│       ├── 1.txt
│       ├── 2.txt
│       └── 3.txt
│
├── testy.ipynb                # Notebook z testami
├── requirements.txt           # Zależności Pythona
└── README.md                  # Ten plik
```

---

## 🔧 Wymagane Komponenty

### Python i Zależności
- **Python 3.9+**
- Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

### Główne Biblioteki
- `langchain` - framework RAG
- `chromadb` - baza wektorowa
- `sentence-transformers` - embeddingi
- `llama-cpp-python` - obsługa modeli GGUF
- `pypdf` - przetwarzanie PDF

---

## 🚀 Instalacja i Uruchomienie

### 1. Instalacja Zależności
```bash
cd lokalnie
pip install -r ../requirements.txt
```

### 2. Przygotowanie Modeli

#### Model Embeddingowy (pobierany automatycznie)
- `sentence-transformers/all-mpnet-base-v2` (~786 MB)
- Pobierany automatycznie przy pierwszym uruchomieniu do cache'u HuggingFace
- Wymaga dostępu do internetu tylko przy pierwszym uruchomieniu

#### Modele LLM (pobierane dynamicznie)
Modele GGUF są pobierane automatycznie z HuggingFace podczas pierwszego użycia. 
Konfiguracja w `modele.json`:

| Model | Rozmiar | Wydajność | Dokładność |
|-------|---------|-----------|-----------|
| Gemma 2 9B | ~5.5 GB | Średnia | Wysoka |
| Qwen 2.5 1.5B | ~1.2 GB | Wysoka | Średnia |
| Llama 3.2 1B | ~1 GB | Wysoka | Średnia |
| Phi 3.5 Mini | ~2.3 GB | Wysoka | Wysoka |

> **Zaleta**: Modele pobierane są tylko raz i cache'owane lokalnie. Nieznacząca objętość repozytorium!

### 3. Uruchomienie Systemu

```bash
# Uruchomienie głównej aplikacji
python main_lokalnie.py

# Uruchomienie notebooka Jupyter
jupyter notebook RAG_TEST/RAG.ipynb
```

---

## 📖 Główne Moduły

### `database.py` - Zarządzanie ChromaDB
```python
from database import (
    init_chroma,           # Inicjalizacja bazy
    load_vectorstore,      # Załadowanie wektorów
    load_pdf_documents,    # Ładowanie PDF
    index_documents_in_batches  # Indeksowanie
)
```

**Funkcje:**
- `init_chroma()` - Tworzenie/połączenie z ChromaDB
- `check_collection_exists()` - Sprawdzenie kolekcji
- `load_pdf_documents()` - Wczytanie dokumentów
- `index_documents_in_batches()` - Indeksowanie z batchami

### `embedding.py` - Modele Embeddingowe
```python
from embedding import get_embeddings
embeddings = get_embeddings()  # all-mpnet-base-v2
```

Model: `sentence-transformers/all-mpnet-base-v2`
- Wymiar: 768
- Optymalizacja: CPU
- Batch size: 8

### `chunks.py` - Chunkowanie Dokumentów
```python
from chunks import smart_chunk_documents
chunks = smart_chunk_documents(documents)
```

**Strategie:**
- `create_optimized_splitter()` - RecursiveCharacterTextSplitter
- `create_token_splitter()` - SentenceTransformersTokenTextSplitter
- Rozmiar chunka: 512 znaków
- Overlap: 50 znaków

### `ask_rag.py` - Logika RAG
```python
from ask_rag import get_llm, ask_rag

llm = get_llm()
response = ask_rag(question, vectorstore, llm)
```

**Cechy:**
- Pobieranie kontekstu z bazy wektorowej
- Tworzenie prompta w języku polskim
- Generowanie odpowiedzi

### `ask_rag__from_json_.py` - Benchmark Modeli
```python
from ask_rag__from_json_ import run_benchmark

run_benchmark(query, vectorstore)
# Porównuje wszystkie modele z modele.json
```

---

## ⚙️ Konfiguracja

### Modele LLM (`modele.json`)

```json
{
  "models": {
    "gemma_2_9b": {
      "repo_id": "bartowski/gemma-2-9b-it-GGUF",
      "filename": "gemma-2-9b-it-IQ2_M.gguf",
      "prompt_template": "..."
    },
    "qwen_2.5_1.5b": {
      "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
      "filename": "qwen2.5-1.5b-instruct-q8_0.gguf",
      "prompt_template": "..."
    }
  }
}
```

### Parametry Modelu (`ask_rag.py`)

```python
MODEL_DIR = "model_llm"
N_THREADS = 4  # Dostosuj do liczby rdzeni CPU
CHUNK_SIZE = 512  # Rozmiar kontekstu
N_CTX = 2048  # Maksymalna długość sekwencji
```

---

## 💡 Przykład Użycia

### Podstawowe Pytanie
```python
from main_lokalnie import main
from embedding import get_embeddings
from database import load_vectorstore, init_chroma
from ask_rag import ask_rag, get_llm

# Inicjalizacja
client = init_chroma()
embeddings = get_embeddings()
vectorstore = load_vectorstore(client, "documents", embeddings)
llm = get_llm()

# Pytanie
query = "Jakie ubezpieczenie uczelnia zaleca studentom?"
response = ask_rag(query, vectorstore, llm)
print(response)
```

### Benchmark Modeli
```python
from ask_rag__from_json_ import run_benchmark

query = "Jakie ubezpieczenie uczelnia zaleca studentom?"
run_benchmark(query, vectorstore)
# Wyniki w results.json
```

---

## 📊 Wyniki i Logi

### Pliki Wynikowe
- `results.json` - Wyniki benchmarku (Format: lista dict z queryami, modelami i odpowiedziami)
- `results_z_pdfow.json` - Wyniki specyficzne dla PDF

### Struktura Wyniku
```json
{
  "query": "Pytanie...",
  "model": "Gemma 2 9B",
  "response": "Odpowiedź...",
  "context": "Kontekst wyszukany z bazy",
  "processing_time": 12.5
}
```

---

## 📋 Dokumenty Testowe

### Dokumenty Dostępne
- **1.pdf / 1.txt** - Dokument akademiczny
- **2.pdf / 2.txt** - Dokument akademiczny
- **3.pdf / 3.txt** - Dokument akademiczny

Umieść dokumenty w:
- `dokumenty/pdfy/` - pliki PDF
- `dokumenty/txt/` - pliki tekstowe

---

## 🔍 Diagnostyka i Troubleshooting

### Problem: Baza wektorowa nie inicjalizuje się
```python
from database import check_collection_exists, init_chroma
client = init_chroma()
check_collection_exists(client, "documents")
```

### Problem: Powolna generacja odpowiedzi
- Zmniejsz `n_threads` w `ask_rag.py`
- Użyj mniejszego modelu (np. Qwen 1.5B zamiast Gemma 9B)
- Zmniejsz `N_CTX` z 2048 do 1024

### Problem: Brak dostępu do internetu przy pobieraniu modeli
```bash
# Modele są cache'owane w HuggingFace cache directory
# Możesz pobrać je ręcznie na innym komputerze z dostępem do sieci:
huggingface-cli download bartowski/gemma-2-9b-it-GGUF gemma-2-9b-it-IQ2_M.gguf --local-dir .

# Lub ustawić ścieżkę cache'u:
export HF_HOME=/ścieżka/do/cache
```

---

## 🧪 Testy

### Notebook Testowy
```bash
jupyter notebook RAG_TEST/RAG.ipynb
```

Zawiera:
- Testy wczytywania dokumentów
- Testy embeddingów
- Testy RAG z różnymi modelami
- Benchmarking wydajności

---

## 📈 Optymalizacja

### Dla Słabszych Komputerów
```python
# ask_rag.py
N_THREADS = 2  # Zmniejsz z 4
N_BATCH = 64   # Zmniejsz z 128

# chunks.py
CHUNK_SIZE = 256  # Zmniejsz z 512
CHUNK_OVERLAP = 25
```

### Dla Silniejszych Komputerów
```python
# ask_rag.py
N_THREADS = 8  # Zwiększ
N_BATCH = 256  # Zwiększ
N_CTX = 4096  # Zwiększ
```

---

## 📚 Dodatkowe Zasoby

### LangChain
- [Dokumentacja RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Integration](https://python.langchain.com/docs/integrations/databases/chroma/)

### Sentence Transformers
- [all-mpnet-base-v2 Model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [Dokumentacja](https://www.sbert.net/)

### Modele GGUF
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [HuggingFace GGUF Models](https://huggingface.co/models?search=gguf)

---

## ⚖️ Licencja

Projekt wykorzystuje:
- **ChromaDB** - Apache 2.0
- **LangChain** - MIT
- **Sentence-Transformers** - Apache 2.0
- **Modele** - różne licencje (sprawdź na HuggingFace)

---

## 🤝 Wsparcie

### Główne Funkcjonalności
- ✅ Indexowanie dokumentów PDF i TXT
- ✅ Wyszukiwanie semantyczne
- ✅ Generowanie odpowiedzi z kontekstem
- ✅ Porównanie wydajności modeli
- ✅ Działanie offline (bez API)
- ✅ Optymalizacja dla CPU

### Możliwości Rozwinięcia
- [ ] Fine-tuning modeli na domenie
- [ ] Interfejs webowy (FastAPI/Flask)
- [ ] API REST
- [ ] Multi-language support
- [ ] Caching odpowiedzi
- [ ] GPU support

---

## 📝 Notatki

- **Środowisko**: Windows/Linux/macOS
- **Python**: 3.9+
- **RAM**: Minimum 4GB (rekomendacja 8GB+)
- **Dysk**: 10GB+ (dla modeli)
- **CPU**: Multi-core rekomendowany

---

**Ostatnia aktualizacja:** 2026-04-11  
**Wersja projektu:** 1.0  
**Status:** Aktywny

