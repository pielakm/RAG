from database import (
    init_chroma,
    load_vectorstore,
    load_pdf_documents,
    index_documents_in_batches,
    check_collection_exists
)
from embedding import get_embeddings
# Importujemy run_benchmark zamiast pojedynczych funkcji
from ask_rag__from_json_ import run_benchmark

# Stałe konfiguracyjne
COLLECTION_NAME = "documents"
DOKUMENTY_PATH = "../dokumenty/txt"


def main():
    # 1. Inicjalizacja klienta bazy
    client = init_chroma()

    # 2. Sprawdzenie/Ładowanie bazy dokumentów
    if check_collection_exists(client, COLLECTION_NAME):
        print("✓ Baza danych gotowa. Ładowanie istniejącej kolekcji...")
        embeddings = get_embeddings()
        vectorstore = load_vectorstore(client, COLLECTION_NAME, embeddings)
    else:
        print("✗ Kolekcja nie istnieje. Tworzenie nowej...")
        docs = load_pdf_documents(DOKUMENTY_PATH)
        if not docs:
            print("Błąd: Brak dokumentów!")
            return

        # Zakładam, że import smart_chunk_documents jest dostępny
        from chunks import smart_chunk_documents
        all_splits = smart_chunk_documents(docs)
        embeddings = get_embeddings()
        vectorstore = index_documents_in_batches(all_splits, embeddings, client, COLLECTION_NAME)

    # 3. Definicja pytania testowego
    query = "Jakie ubezpieczenie uczelnia zaleca studentom?"
    print(f"\n🚀 URUCHAMIAM TEST PORÓWNAWCZY DLA PYTANIA:")
    print(f"👉 {query}\n")

    # 4. Wywołanie benchmarku (pętla po wszystkich modelach z JSON)
    # Ta funkcja sama zajmie się ładowaniem, pytaniem i czyszczeniem RAM
    run_benchmark(query, vectorstore)

    print("\n📊 Wyniki zostały zapisane do pliku results.json")


if __name__ == "__main__":
    main()