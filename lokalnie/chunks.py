from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

# === OPTYMALIZACJA DLA TWOJEGO LAPTOPA ===
CHUNK_SIZE = 512  # Zmniejszono na 512 aby pasować do limitu modelu embedding BGE
CHUNK_OVERLAP = 50


# === FUNKCJE CHUNKOWANIA ===
def create_optimized_splitter():
    """Splitter zoptymalizowany pod polski język akademicki"""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",  # Akapity
            "\n",  # Linie
            ". ",  # Zdania
            "! ",  # Wykrzykniki
            "? ",  # Pytania
            "; ",  # Średniki
            ", ",  # Przecinki
            " ",  # Spacje
            ""
        ]
    )


def create_token_splitter():
    """Splitter bazujący na tokenach - dokładniejszy"""
    return SentenceTransformersTokenTextSplitter(
        chunk_overlap=CHUNK_OVERLAP,
        tokens_per_chunk=256,  # Dla mpnet-base to ~350 znaków
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


def smart_chunk_documents(docs):
    """Inteligentne chunkowanie z diagnostyką"""
    print("--- Chunkowanie dokumentów ---")

    if not docs:
        print("Brak dokumentów do przetworzenia")
        return []

    total_chars = sum(len(doc.page_content) for doc in docs)
    avg_doc_length = total_chars / len(docs)

    print(f"Łączna liczba dokumentów: {len(docs)}")
    print(f"Średnia długość dokumentu: {avg_doc_length:.0f} znaków")

    # Wybór strategii chunkowania
    if avg_doc_length > 1000:
        print("✓ Wybieram strategię dla długich dokumentów akademickich")
        splitter = create_optimized_splitter()
    else:
        print("✓ Wybieram strategię token-based dla lepszej precyzji")
        splitter = create_token_splitter()

    chunks = splitter.split_documents(docs)

    # Diagnostyka wyników
    if chunks:
        avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
        print(f"Utworzono {len(chunks)} fragmentów")
        print(f"Średnia długość fragmentu: {avg_chunk_size:.0f} znaków")
    else:
        raise ValueError("Błąd: Nie udało się utworzyć fragmentów dokumentów")

    return chunks
