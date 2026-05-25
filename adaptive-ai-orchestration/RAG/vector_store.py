# rag/vector_store.py

import faiss
import numpy as np
import pickle
import os

# ──────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────

FAISS_INDEX_PATH    = "faiss_index/index.faiss"
METADATA_PATH       = "faiss_index/metadata.pkl"
EMBEDDING_DIMENSION = 1024  # Titan V2 output size


def build_index(embedded_chunks: list[dict]) -> None:
    """
    Builds a FAISS index from embedded chunks and saves to disk.

    Args:
        embedded_chunks: list of dicts with text, source,
                         chunk_id, embedding
    """
    os.makedirs("faiss_index", exist_ok=True)

    # Build embedding matrix
    vectors  = np.array(
        [chunk["embedding"] for chunk in embedded_chunks],
        dtype=np.float32
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)

    # Create FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(vectors)

    # Save index to disk
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata separately
    metadata = [
        {
            "text":     chunk["text"],
            "source":   chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in embedded_chunks
    ]

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index built — {len(embedded_chunks)} vectors stored")
    print(f"   Index saved to: {FAISS_INDEX_PATH}")
    print(f"   Metadata saved to: {METADATA_PATH}")


def load_index():
    """
    Loads FAISS index and metadata from disk.

    Returns:
        tuple of (faiss_index, metadata_list)
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            f"Run build_index first."
        )

    index    = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def search_index(query_embedding: list[float],
                 top_k: int = 5) -> list[dict]:
    """
    Searches FAISS index for most similar chunks.

    Args:
        query_embedding: 1024-dim vector from Titan
        top_k:           number of results to return

    Returns:
        list of dicts with text, source, score
    """
    index, metadata = load_index()

    # Prepare query vector
    query_vector = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # Search
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text":   metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "score":  round(float(score), 4)
        })

    return results


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    from RAG.document_loader import load_all_documents
    from RAG.embedder import embed_chunks, embed_query

    print("\n── Vector Store Test ──\n")

    # Load and embed all chunks
    chunks         = load_all_documents()
    embedded       = embed_chunks(chunks)

    # Build index
    build_index(embedded)

    # Test search
    print("\nTesting search...")
    query          = "What is the leave policy?"
    query_embedding = embed_query(query)
    results        = search_index(query_embedding, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"   Source: {result['source']}")
        print(f"   Score:  {result['score']}")
        print(f"   Text:   {result['text'][:150]}...")
        print()

    print("✅ Vector store working correctly!")