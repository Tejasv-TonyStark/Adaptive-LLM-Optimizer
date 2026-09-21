"""Versioned JSON metadata and atomic manifest publication; legacy pickle is never loaded."""
import json
import os
import uuid
from pathlib import Path
from functools import lru_cache
import faiss
import numpy as np
DIRECTORY = Path(__file__).resolve().parents[1] / "faiss_index"
EMBEDDING_DIMENSION = 1024
def build_index(embedded_chunks):
    if not embedded_chunks:
        raise ValueError("Cannot build an empty index")
    vectors = np.asarray([c["embedding"] for c in embedded_chunks], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSION or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding dimensions or values")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Zero document vector")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(vectors)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    version = uuid.uuid4().hex
    faiss.write_index(index, str(DIRECTORY / f"{version}.faiss"))
    metadata = [{k: c[k] for k in ("text", "source", "chunk_id", "page")} for c in embedded_chunks]
    (DIRECTORY / f"{version}.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = dict(version=version, dimension=EMBEDDING_DIMENSION,
                    embedding_model="amazon.titan-embed-text-v2:0", count=len(metadata))
    temp = DIRECTORY / f"{version}.manifest.tmp"
    temp.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(temp, DIRECTORY / "manifest.json")
    _load_version.cache_clear()
@lru_cache(maxsize=2)
def _load_version(directory, version):
    if len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("Invalid index version")
    path = Path(directory)
    index = faiss.read_index(str(path / f"{version}.faiss"))
    metadata = json.loads((path / f"{version}.json").read_text(encoding="utf-8"))
    if index.ntotal != len(metadata) or index.d != EMBEDDING_DIMENSION:
        raise ValueError("Index and metadata do not match")
    return index, metadata
def load_index():
    manifest_path = DIRECTORY / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Build the JSON index with python -m RAG.vector_store; legacy pickle is unsupported.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["embedding_model"] != "amazon.titan-embed-text-v2:0":
        raise ValueError("Embedding model mismatch; rebuild index")
    index, metadata = _load_version(str(DIRECTORY), manifest["version"])
    if manifest["count"] != len(metadata) or manifest["dimension"] != index.d:
        raise ValueError("Manifest does not match index")
    return index, metadata
def search_index(query_embedding, top_k=5, allowed_sources=None):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    index, metadata = load_index()
    vector = np.asarray([query_embedding], dtype=np.float32)
    if vector.shape != (1, index.d) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
        raise ValueError("Invalid query embedding")
    faiss.normalize_L2(vector)
    # Filter before selecting top-k so inaccessible neighbors do not hide valid hits.
    count = index.ntotal if allowed_sources is not None else min(top_k, index.ntotal)
    scores, indices = index.search(vector, count)
    return [{**metadata[int(idx)], "score": float(score)}
            for score, idx in zip(scores[0], indices[0]) if idx >= 0
            and (allowed_sources is None or metadata[int(idx)]["source"] in allowed_sources)][:top_k]
if __name__ == "__main__":
    from RAG.document_loader import load_all_documents
    from RAG.embedder import embed_chunks
    chunks = load_all_documents()
    build_index(embed_chunks(chunks))
    print(f"Published {len(chunks)} chunks. Ingestion calls incur embedding charges separately from request usage.")
