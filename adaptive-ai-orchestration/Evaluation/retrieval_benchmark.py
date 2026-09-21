"""Measure supporting-passage recall and empty-retrieval behavior separately.

--live makes paid embedding calls against the installed index, never generation.
--observations replays a JSON mapping from case id to {context, chunks, found}.
The small provided cases are development fixtures, not independent validation.
"""
import argparse
import json
from pathlib import Path

def normalize(text):
    return " ".join(text.casefold().split())

def retrieval_report(cases, retrieve_fn):
    results = []
    for case in cases:
        retrieved = retrieve_fn(case)
        context = normalize(retrieved["context"])
        expected = case.get("supporting_passages", [])
        hits = sum(normalize(p) in context for p in expected)
        results.append(dict(id=case["id"], expected_passages=len(expected), retrieved_passages=hits,
                            passage_recall=hits/len(expected) if expected else None,
                            expected_abstention=case.get("expect_abstention", False),
                            empty_retrieval=not retrieved["found"], observation=retrieved))
    total = sum(r["expected_passages"] for r in results)
    unsupported = [r for r in results if r["expected_abstention"]]
    return dict(kind="retrieval development evaluation; source support, not answer quality",
                passage_recall=sum(r["retrieved_passages"] for r in results)/total if total else None,
                empty_retrieval_on_unanswerable=sum(r["empty_retrieval"] for r in unsupported)/len(unsupported) if unsupported else None,
                results=results)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live",action="store_true")
    mode.add_argument("--observations",type=Path)
    parser.add_argument("--cases",type=Path,default=Path(__file__).with_name("retrieval_cases.json"))
    parser.add_argument("--output",type=Path,default=Path("benchmark_results/retrieval.json"))
    parser.add_argument("--threshold",type=float,default=.75)
    parser.add_argument("--top-k",type=int,default=5)
    parser.add_argument("--user-id",type=int,default=1)
    args=parser.parse_args()
    cases=json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    usage=[]
    if args.live:
        from RAG.retriever import retrieve
        fetch=lambda case: retrieve(case["query"],top_k=args.top_k,user_id=args.user_id,
                                    threshold=args.threshold,usage=usage)
    else:
        observations=json.loads(args.observations.read_text(encoding="utf-8"))
        fetch=lambda case: observations[case["id"]]
    report=retrieval_report(cases,fetch)
    report.update(threshold=args.threshold,top_k=args.top_k,usage=usage)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Report: {args.output}; passage recall: {report['passage_recall']}")

if __name__=="__main__":
    main()
