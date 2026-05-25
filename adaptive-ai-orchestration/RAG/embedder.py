# rag/embedder.py

from Execution.Bedrock_client import get_embedding


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generates embeddings for all document chunks
    using Titan Text Embeddings V2.

    Args:
        chunks: list of dicts from document_loader
                each with text, source, chunk_id

    Returns:
        list of dicts with embedding added
    """
    embedded = []
    total    = len(chunks)

    print(f"Embedding {total} chunks...")

    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk["text"])

            embedded.append({
                "text":      chunk["text"],
                "source":    chunk["source"],
                "chunk_id":  chunk["chunk_id"],
                "embedding": embedding
            })

            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{total}")

        except Exception as e:
            print(f"❌ Failed to embed chunk {chunk['chunk_id']}: {e}")
            continue

    print(f"✅ Embedded {len(embedded)}/{total} chunks successfully")
    return embedded


def embed_query(query: str) -> list[float]:
    """
    Generates embedding for a single query.
    Used at retrieval time to find similar chunks.

    Args:
        query: user question string

    Returns:
        list of floats — 1024 dimension vector
    """
    return get_embedding(query)


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    from RAG.document_loader import load_all_documents

    print("\n── Embedder Test ──\n")

    chunks   = load_all_documents()

    if not chunks:
        print("❌ No chunks to embed")
        exit()

    # Test with first 3 chunks only to save credits
    test_chunks = chunks[:3]
    embedded    = embed_chunks(test_chunks)

    print(f"\nSample embedding:")
    print(f"Chunk:          {embedded[0]['chunk_id']}")
    print(f"Text:           {embedded[0]['text'][:100]}...")
    print(f"Embedding size: {len(embedded[0]['embedding'])}")
    print("\n✅ Embedder working correctly!")
    