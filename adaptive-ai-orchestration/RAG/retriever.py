# rag/retriever.py

from RAG.embedder import embed_query
from RAG.vector_store import search_index

# ──────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.70  # minimum score to include a chunk
MAX_CONTEXT_CHUNKS   = 3     # max chunks to include in prompt


def retrieve(query: str, top_k: int = 5) -> dict:
    """
    Main retrieval function.
    Embeds the query, searches FAISS, filters by threshold.

    Args:
        query: user question
        top_k: number of candidates to fetch before filtering

    Returns:
        dict with context string and source chunks
    """
    # Embed the query
    query_embedding = embed_query(query)

    # Search FAISS index
    results = search_index(query_embedding, top_k=top_k)

    # Filter by similarity threshold
    filtered = [
        r for r in results
        if r["score"] >= SIMILARITY_THRESHOLD
    ]

    # If nothing passes threshold use top 2 anyway
    if not filtered:
        filtered = results[:2]

    # Limit to max chunks
    filtered = filtered[:MAX_CONTEXT_CHUNKS]

    # Build context string for prompt
    context_parts = []
    for i, chunk in enumerate(filtered):
        context_parts.append(
            f"[Source: {chunk['source']} | Score: {chunk['score']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return {
        "context":  context,
        "chunks":   filtered,
        "found":    len(filtered) > 0
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Retriever Test ──\n")

    test_queries = [
        "What is the leave policy?",
        "What is the notice period for resignation?",
        "What are the salary benefits?",
        "What is Python?"  # general query — should return low scores
    ]

    for query in test_queries:
        print(f"Query:   {query}")
        result = retrieve(query)
        print(f"Found:   {result['found']}")
        print(f"Chunks:  {len(result['chunks'])}")
        if result["chunks"]:
            print(f"Top score: {result['chunks'][0]['score']}")
            print(f"Source:    {result['chunks'][0]['source']}")
            print(f"Text:      {result['chunks'][0]['text'][:100]}...")
        print()

    print("✅ Retriever working correctly!")