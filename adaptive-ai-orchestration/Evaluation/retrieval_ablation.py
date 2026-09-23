"""Compare retrieval variants on fixed labeled supporting-passage cases.

This makes embedding calls when run.  It measures source-passage recall only;
it does not measure generated-answer quality, latency, or production accuracy.
The supplied cases are development fixtures and must not be presented as an
independent benchmark.
"""
import argparse
import json
from pathlib import Path


def normalize(text):
    return " ".join(text.casefold().split())


def selected_chunks(query, candidates, variant, allowed_sources, top_k):
    """Return the chunks each retrieval design would expose to the answerer."""
    from RAG.retriever import _add_neighbor_chunks, _rerank

    seed_count = max(1, top_k // 2)
    if variant == "vector":
        return candidates[:top_k]
    reranked = _rerank(query, candidates)
    if variant == "rerank":
        return reranked[:top_k]
    if variant == "rerank_neighbors":
        return _add_neighbor_chunks(reranked[:seed_count], allowed_sources)
    raise ValueError(f"Unknown variant: {variant}")


def evaluate(cases, embeddings, top_k, threshold):
    from RAG.access import allowed_sources
    from RAG.vector_store import search_index

    variants = ("vector", "rerank", "rerank_neighbors")
    report = {name: [] for name in variants}
    permissions = allowed_sources(1)
    for case in cases:
        candidates = search_index(embeddings[case["id"]], 20, allowed_sources=permissions)
        for variant in variants:
            chunks = [chunk for chunk in selected_chunks(case["query"], candidates, variant,
                                                          permissions, top_k)
                      if chunk["score"] >= threshold]
            context = normalize("\n".join(chunk["text"] for chunk in chunks))
            passages = [normalize(passage) for passage in case.get("supporting_passages", [])]
            hits = sum(passage in context for passage in passages)
            report[variant].append({
                "id": case["id"],
                "supporting_passages": len(passages),
                "retrieved_passages": hits,
                "passage_recall": hits / len(passages) if passages else None,
                "expected_abstention": case.get("expect_abstention", False),
                "empty_retrieval": not chunks,
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
            })
    summaries = {}
    for variant, rows in report.items():
        total = sum(row["supporting_passages"] for row in rows)
        found = sum(row["retrieved_passages"] for row in rows)
        unanswerable = [row for row in rows if row["expected_abstention"]]
        summaries[variant] = {
            "passage_recall": found / total if total else None,
            "supporting_passages_found": found,
            "supporting_passages_total": total,
            "empty_retrieval_on_unanswerable": (
                sum(row["empty_retrieval"] for row in unanswerable) / len(unanswerable)
                if unanswerable else None
            ),
            "results": rows,
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("retrieval_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/retrieval_ablation.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.top_k < 1 or not -1 <= args.threshold <= 1:
        raise ValueError("top-k must be positive and threshold must be in [-1, 1]")

    from Execution.Bedrock_client import get_embedding_result

    case_document = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = case_document["cases"]
    embeddings = {case["id"]: get_embedding_result(case["query"])["embedding"] for case in cases}
    result = {
        "kind": "retrieval ablation; source-passage recall only",
        "limitations": case_document.get(
            "description",
            f"{len(cases)} author-labeled cases; not independent evidence for a resume improvement claim.",
        ),
        "top_k": args.top_k,
        "threshold": args.threshold,
        "variants": evaluate(cases, embeddings, args.top_k, args.threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Report: {args.output}")
    for name, summary in result["variants"].items():
        print(f"{name}: {summary['supporting_passages_found']}/{summary['supporting_passages_total']} "
              f"supporting passages ({summary['passage_recall']:.0%})")


if __name__ == "__main__":
    main()
