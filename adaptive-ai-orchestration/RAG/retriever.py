import os
import time
from Execution.Bedrock_client import get_embedding_result
from RAG.vector_store import search_index, load_index
from tracking.usage import usage_record
from Execution.prompt_builder import MAX_RAG_CONTEXT_CHARS
from RAG.access import allowed_sources
# Titan embeddings for the small handbook chunks in the shipped index score well
# below 0.75 even for direct policy questions (for example, working-hours lookup
# scores about 0.63).  Keep a low candidate threshold and rely on the strict
# extractive-evidence check before returning an answer to the user.
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.05"))
MAX_CONTEXT_CHUNKS = 5
RERANK_CANDIDATES = 20
RETRIEVAL_STOPWORDS = {"a", "an", "and", "are", "about", "for", "how", "is", "of", "our",
                       "policy", "the", "to", "what", "when", "where", "which", "who", "why"}
if not -1 <= SIMILARITY_THRESHOLD <= 1:
    raise ValueError("RAG_SIMILARITY_THRESHOLD must be in [-1, 1]")

def _terms(text):
    import re
    return {word for word in re.findall(r"[a-z0-9]{3,}", text.casefold())
            if word not in RETRIEVAL_STOPWORDS}

def _rerank(query, candidates):
    """Blend vector similarity with exact topical-term overlap."""
    terms = _terms(query)
    def score(chunk):
        overlap = len(terms & _terms(chunk["text"])) / max(len(terms), 1)
        return (.7 * chunk["score"] + .3 * overlap, chunk["score"])
    return sorted(candidates, key=score, reverse=True)

def _add_neighbor_chunks(chunks, allowed_sources):
    """Keep adjacent PDF chunks together so FAQ prompts retain their answer."""
    _, metadata = load_index()
    selected = {(c["source"], c["chunk_id"]) for c in chunks}
    expanded = list(chunks)
    for chunk in chunks:
        prefix, _, number = chunk["chunk_id"].rpartition(":c")
        if not number.isdigit():
            continue
        wanted = f"{prefix}:c{int(number)+1}"
        neighbor = next((item for item in metadata if item["chunk_id"] == wanted
                         and item["source"] == chunk["source"]
                         and item["page"] == chunk["page"]
                         and (allowed_sources is None or item["source"] in allowed_sources)), None)
        if neighbor and (neighbor["source"], neighbor["chunk_id"]) not in selected:
            expanded.append({**neighbor, "score": chunk["score"]})
            selected.add((neighbor["source"], neighbor["chunk_id"]))
        if len(expanded) >= MAX_CONTEXT_CHUNKS:
            break
    return expanded[:MAX_CONTEXT_CHUNKS]
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
    candidates = search_index(embedding["embedding"], max(top_k, RERANK_CANDIDATES), allowed_sources=permissions)
    # Reserve half of the context budget for adjacent chunks. This preserves
    # answer text that immediately follows a matched heading or FAQ question.
    seed_count = max(1, top_k // 2)
    results = _add_neighbor_chunks(_rerank(query, candidates)[:seed_count], permissions)
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
