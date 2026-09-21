import os
import time
from Execution.Bedrock_client import get_embedding_result
from RAG.vector_store import search_index
from tracking.usage import usage_record
from Execution.prompt_builder import MAX_RAG_CONTEXT_CHARS
from RAG.access import allowed_sources
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.75"))
MAX_CONTEXT_CHUNKS = 5
if not -1 <= SIMILARITY_THRESHOLD <= 1:
    raise ValueError("RAG_SIMILARITY_THRESHOLD must be in [-1, 1]")
def retrieve(query, top_k=5, usage=None, user_id=None, threshold=None):
    threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
    if not -1 <= threshold <= 1 or top_k < 1:
        raise ValueError("Invalid retrieval threshold or top_k")
    permissions = allowed_sources(user_id)
    if permissions == set():
        return dict(context="", chunks=[], found=False)
    records = usage if usage is not None else []
    started = time.perf_counter()
    try:
        embedding = get_embedding_result(query)
        records.append(usage_record("titan-embed-v2", "embedding", query, embedding,
                                    round((time.perf_counter()-started)*1000)))
    except Exception as exc:
        records.append(usage_record("titan-embed-v2", "embedding", query, {},
                                    round((time.perf_counter()-started)*1000), error=exc))
        raise
    results = search_index(embedding["embedding"], top_k, allowed_sources=permissions)
    chunks, parts = [], []
    for chunk in results:
        if chunk["score"] < threshold or len(chunks) >= MAX_CONTEXT_CHUNKS:
            continue
        part = f"[{len(chunks)+1}] {chunk['source']}, page {chunk['page']}\n{chunk['text']}"
        if len("\n\n---\n\n".join([*parts, part])) > MAX_RAG_CONTEXT_CHARS:
            break
        chunks.append(chunk)
        parts.append(part)
    return dict(context="\n\n---\n\n".join(parts), chunks=chunks, found=bool(chunks))
