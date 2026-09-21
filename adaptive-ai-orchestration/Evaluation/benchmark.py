"""Offline routing checks and explicitly requested paid end-to-end experiments."""
import argparse
import json
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
DATASET = Path(__file__).with_name("dataset.json")

def classification_report(cases):
    from core.orchestrator import orchestrate
    results=[]
    for case in cases:
        got=orchestrate(case["query"])
        fields=("intent","complexity","retrieval_needed")
        results.append(dict(id=case["id"], expected={k:case[k] for k in fields},
                            actual={k:got[k] for k in fields},
                            passed=all(got[k]==case[k] for k in fields)))
    return dict(kind="offline classification; no model quality or cost claims",
                total=len(results), passed=sum(r["passed"] for r in results), results=results)

def percentile(values, percent):
    if not values:
        return None
    ordered=sorted(values)
    position=(len(ordered)-1)*percent
    lo=int(position)
    hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)

def live_policy(cases, policy):
    # This is an isolated local experiment; never seed, reset, or learn in the application DB.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database.connection import Base
    from database.models import User, Query, Evaluation, Usage, Probability
    from database.seed import seed_probabilities
    from core.service import process_query, PipelineUnavailable
    from Evaluation.worker import process_one
    results=[]
    with tempfile.TemporaryDirectory() as directory:
        engine=create_engine("sqlite:///"+str(Path(directory)/"benchmark.sqlite"))
        factory=sessionmaker(bind=engine,expire_on_commit=False)
        Base.metadata.create_all(engine)
        with factory() as db:
            db.add(User(id=1,username="benchmark",email="benchmark@local.example",
                        hashed_password="unusable",is_active=True))
            seed_probabilities(db)
            db.commit()
        train=[c for c in cases if c["split"]=="train"]
        heldout=[c for c in cases if c["split"]=="heldout"]
        random.Random(42).shuffle(heldout)
        for case in train+heldout:
            began=time.perf_counter()
            with factory() as db:
                try:
                    row=process_query(db,1,case["query"],"benchmark-"+case["id"],policy="adaptive" if policy=="adaptive" else "static",
                        force_model={"cheapest":"nova-micro","middle":"llama3-8b","strongest":"llama3-70b"}.get(policy),
                        evaluation_rate=1 if case["split"]=="train" else 0,allow_fallback=False)
                    query_id=row.id
                except PipelineUnavailable as exc:
                    query_id=exc.query_id
            if case["split"]=="train":
                # Train each isolated router only on the train partition.
                process_one(factory)
            else:
                # Score held-out answers without updating routing metrics.
                from Evaluation.Evaluator import evaluate_response
                with factory() as db:
                    row=db.get(Query,query_id)
                    if row.status=="completed":
                        score=evaluate_response(row.query_text,row.response,row.context)
                        db.add_all([Usage(query_id=row.id,**u) for u in score["usage"]])
                        if score["success"]:
                            db.add(Evaluation(query_id=row.id,**{k:score[k] for k in (
                                "relevance","correctness","completeness","quality_score","reasoning",
                                "hallucination_flags","retrieval_score","retrieval_warning")}))
                        db.commit()
            elapsed_ms=round((time.perf_counter()-began)*1000)
            with factory() as db:
                row=db.get(Query,query_id)
                ev=db.query(Evaluation).filter_by(query_id=query_id).first()
                usage=db.query(Usage).filter_by(query_id=query_id).all()
                answer=(row.response or "").casefold()
                checks=[p.casefold() in answer for p in case.get("must_contain",[])]
                checks += [p.casefold() not in answer for p in case.get("must_not_contain",[])]
                if case.get("expect_abstention"):
                    checks.append(row.status=="abstained")
                results.append(dict(id=case["id"],split=case["split"],query=case["query"],
                    response=row.response,status=row.status,model=row.model_used,sources=row.sources,
                    quality=ev.quality_score if ev else None,
                    reference_checks_pass=all(checks) if checks else None,
                    retrieval_found=bool(row.context) if case["retrieval_needed"] else None,
                    latency_ms=row.latency_ms,total_with_judge_ms=elapsed_ms,
                    cost_by_stage={s:sum(u.estimated_cost or 0 for u in usage if u.stage==s)
                                   for s in ("generation","embedding","judge")},
                    unknown_cost_attempts=sum(u.estimated_cost is None for u in usage)))
        with factory() as db:
            learned_updates=sum(p.sample_count for p in db.query(Probability).all())
        engine.dispose()
    heldout=[r for r in results if r["split"]=="heldout"]
    qualities=[r["quality"] for r in heldout if r["quality"] is not None]
    checks=[r["reference_checks_pass"] for r in heldout if r["reference_checks_pass"] is not None]
    return dict(policy=policy,heldout_queries=len(heldout),
        learned_updates=learned_updates,
        learning_note="Updates require an independent, current judge calibration report; otherwise seed assumptions remain.",
        abstention_rate=sum(r["status"]=="abstained" for r in heldout)/max(len(heldout),1),
        average_quality=sum(qualities)/len(qualities) if qualities else None,
        judged_count=len(qualities),
        reference_pass_rate=sum(checks)/len(checks) if checks else None,
        failure_rate=sum(r["status"]=="failed" for r in heldout)/max(len(heldout),1),
        median_latency_ms=percentile([r["latency_ms"] for r in heldout],.5),
        p95_latency_ms=percentile([r["latency_ms"] for r in heldout],.95),
        heldout_cost_by_stage={s:sum(r["cost_by_stage"][s] for r in heldout)
                              for s in ("generation","embedding","judge")},
        training_cost=sum(sum(r["cost_by_stage"].values()) for r in results if r["split"]=="train"),
        unknown_cost_attempts=sum(r["unknown_cost_attempts"] for r in results),results=results)

def main(default_suite="all"):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true",help="Make paid Bedrock generation, embedding, and judge calls")
    parser.add_argument("--suite",choices=("all","rag"),default=default_suite)
    parser.add_argument("--dataset",type=Path,default=DATASET)
    parser.add_argument("--output",type=Path,default=Path("benchmark_results/report.json"))
    args=parser.parse_args()
    document=json.loads(args.dataset.read_text(encoding="utf-8"))
    cases=[c for c in document["cases"] if args.suite=="all" or c["category"]=="rag"]
    if args.live:
        report=dict(kind="live held-out comparison",dataset_version=document["version"],
                    limitations=document["description"],
                    policies=[live_policy(cases,p) for p in ("cheapest","middle","strongest","static","adaptive")])
    else:
        report=classification_report(cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Report: {args.output}")
    if not args.live:
        print(f'{report["passed"]}/{report["total"]} labeled routing cases passed.')
        raise SystemExit(0 if report["passed"]==report["total"] else 1)
if __name__=="__main__":
    main()
