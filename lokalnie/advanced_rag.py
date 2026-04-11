import os
import ollama
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from chunks import create_optimized_splitter

# === KONFIGURACJA MODELI (Ollama) ===
# Uwaga: Ollama musi mieć dostęp do tych modeli.
# Jeśli ich nie masz, wykonaj w terminalu: ollama run hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
DOKUMENTY_PATH = "../dokumenty/txt"

# Nasza prosta baza wektorowa w RAM
VECTOR_DB = []


def load_and_chunk_documents(path):
    """Wczytuje pliki i dzieli na fragmenty używając Twojego splittera"""
    all_text = ""
    if not os.path.exists(path):
        print(f"BŁĄD: Folder {path} nie istnieje!")
        return []

    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n\n"

    splitter = create_optimized_splitter()
    chunks = splitter.split_text(all_text)
    return chunks


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if not norm_a or not norm_b: return 0
    return dot_product / (norm_a * norm_b)


def build_database(chunks):
    print(f"--- Budowanie bazy dla {len(chunks)} fragmentów ---")
    for i, chunk in enumerate(chunks):
        try:
            # Generowanie embeddingu przez Ollama
            response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
            embedding = response['embeddings'][0]
            VECTOR_DB.append((chunk, embedding))
        except Exception as e:
            # Jeśli fragment jest za długi, podziel go na połowy
            if "exceeds the context length" in str(e):
                print(f"\n⚠️  Fragment {i+1} za długi ({len(chunk)} znaków), dzielę go...")
                mid = len(chunk) // 2
                # Spróbuj znaleźć ostatnią spację dla naturalnego podziału
                for j in range(mid, max(0, mid-100), -1):
                    if chunk[j] == ' ':
                        mid = j
                        break
                chunk1, chunk2 = chunk[:mid], chunk[mid:]
                
                # Rekurencyjnie przetwórz podzielone fragmenty
                for sub_chunk in [chunk1.strip(), chunk2.strip()]:
                    if sub_chunk:
                        try:
                            response = ollama.embed(model=EMBEDDING_MODEL, input=sub_chunk)
                            embedding = response['embeddings'][0]
                            VECTOR_DB.append((sub_chunk, embedding))
                        except Exception as e2:
                            print(f"❌ Nie mogę zaembedować podgrupy: {str(e2)}")
            else:
                print(f"❌ Błąd przy przetwarzaniu fragmentu {i+1}: {str(e)}")
        
        if (i + 1) % 10 == 0:
            print(f"Przetworzono {i + 1}/{len(chunks)} fragmentów...")


def retrieve(query, top_n=3):
    """Wyszukiwanie kontekstu"""
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def plot_similarity_heatmap(query_embedding, doc_embeddings, labels):
    """Tworzy wykres ciepła zależności między pytaniem a fragmentami"""
    # Łączymy embedding pytania z dokumentami do jednej macierzy
    all_embeddings = np.vstack([query_embedding, doc_embeddings])
    sim_matrix = sk_cosine_similarity(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, cmap='YlGnBu', 
                xticklabels=['Pytanie'] + labels, 
                yticklabels=['Pytanie'] + labels,
                fmt='.2f')
    plt.title("Macierz Podobieństwa Semantycznego (Cosine Similarity)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_top_scores(similarities):
    """Wykres słupkowy najlepszych dopasowań"""
    # Sortujemy dane do wykresu
    similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_n = similarities_sorted[:5]  # top 5
    
    scores = [s[1] for s in top_n]
    labels = [f"Chunk {i}" for i in range(len(top_n))]
    
    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(scores)) for i in range(len(scores))]
    plt.bar(labels, scores, color=colors)
    plt.axhline(y=0.7, color='r', linestyle='--', label='Próg istotności (0.7)')
    plt.ylabel("Score Podobieństwa")
    plt.title("Top 5 najbardziej trafnych fragmentów")
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():
    # 1. Przygotowanie danych
    dataset = load_and_chunk_documents(DOKUMENTY_PATH)
    if not dataset: return

    # 2. Budowanie bazy wektorowej (Eembeddingi)
    build_database(dataset)

    # 3. Pytanie i Embedding
    input_query = "Jakie ubezpieczenie uczelnia zaleca studentom?"
    print(f"\nUżytkownik: {input_query}")

    # Przygotowanie embeddingów do wizualizacji
    all_doc_embeddings = np.array([item[1] for item in VECTOR_DB])
    chunk_labels = [f"D{i}" for i in range(len(VECTOR_DB))]

    query_resp = ollama.embed(model=EMBEDDING_MODEL, input=input_query)
    query_embedding = np.array(query_resp['embeddings'][0]).reshape(1, -1)

    # 4. Obliczanie podobieństwa przez scikit-learn (szybsze)
    sims = sk_cosine_similarity(query_embedding, all_doc_embeddings)[0]
    results = list(zip([item[0] for item in VECTOR_DB], sims))

    print('\nZnaleziony kontekst (top 3):')
    retrieved_knowledge = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    for chunk, similarity in retrieved_knowledge:
        print(f' - (podob: {similarity:.2f}) {chunk[:100]}...')

    # --- WIZUALIZACJA I METRYKI ---
    print("\n--- Generowanie wykresów metryk ---")
    plot_top_scores(results)
    # Dla czytelności heatmapy bierzemy tylko pierwsze 10 chunków
    if len(all_doc_embeddings) > 10:
        plot_similarity_heatmap(query_embedding, all_doc_embeddings[:10], chunk_labels[:10])
    else:
        plot_similarity_heatmap(query_embedding, all_doc_embeddings, chunk_labels)

    # 5. Generowanie odpowiedzi przez LLM
    context_text = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])

    instruction_prompt = f'''Jesteś pomocnym asystentem akademickim.
Użyj wyłącznie poniższego kontekstu, aby odpowiedzieć na pytanie.
Jeśli w kontekście nie ma odpowiedzi, powiedz że nie wiesz.
Odpowiadaj po polsku.

KONTEKST:
{context_text}
'''

    print('\nOdpowiedź Chatbota:')
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)


if __name__ == "__main__":
    main()