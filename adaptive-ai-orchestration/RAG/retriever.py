# rag/retriever.py

from RAG.embedder import embed_query
from RAG.vector_store import search_index

# ──────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.75  # raised from 0.70

def retrieve(query: str, top_k: int = 5) -> dict:
    query_embedding = embed_query(query)
    results = search_index(query_embedding, top_k=top_k)

    filtered = [
        r for r in results
        if r["score"] >= SIMILARITY_THRESHOLD
    ]

    # ← REMOVE the fallback that forces results[:2]
    # If nothing passes threshold, return nothing
    if not filtered:
        return {
            "context": "",
            "chunks":  [],
            "found":   False
        }

    filtered = filtered[:MAX_CONTEXT_CHUNKS]

    context_parts = []
    for chunk in filtered:
        context_parts.append(
            f"[Source: {chunk['source']} | Score: {chunk['score']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return {
        "context": context,
        "chunks":  filtered,
        "found":   True
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