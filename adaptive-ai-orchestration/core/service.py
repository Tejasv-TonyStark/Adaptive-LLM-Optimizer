"""One request pipeline shared by the API and the benchmark runner."""
import hashlib
import time
from database.models import Query, Usage, AuditLog
from core.orchestrator import orchestrate
from core.decision_engine import select_best_model, SUITABLE_MODELS, STATIC_MODELS
from core.model_health import model_health
from core.conversation import resolve_query
from core.config import EVALUATION_SAMPLE_RATE
from Execution.Execution_layer import execute_query, ExecutionError, get_output_token_budget
from Execution.prompt_builder import build_prompt
from RAG.retriever import retrieve
from Security.Security_Guard import inspect_query
from tracking.usage import estimate_tokens

class QueryRejected(ValueError):
    pass
class PipelineUnavailable(RuntimeError):
    def __init__(self, query_id):
        self.query_id = query_id
        super().__init__("A required service is unavailable; please retry later.")

def process_query(db, user_id, query, session_id, policy=None, force_model=None,
                  evaluation_rate=None, allow_fallback=True):
    start = time.perf_counter()
    security = inspect_query(query)
    if not security["safe"]:
        raise QueryRejected(security["reason"])
    clean = security["clean_query"]
    conversation = resolve_query(db, user_id, session_id, clean)
    resolved = conversation["resolved"]
    routing = orchestrate(resolved)
    scoped = inspect_query(clean, retrieval_needed=routing["retrieval_needed"])
    if not scoped["safe"]:
        raise QueryRejected(scoped["reason"])
    routing["conversation"] = dict(resolved=resolved, followup=conversation["followup"],
                                    topic=conversation.get("topic", clean))
    usage, sources, context, source_chunks = [], [], None, []
    selected, decision = force_model, {}
    try:
        if routing["retrieval_needed"] and not conversation["needs_clarification"]:
            retrieved = retrieve(resolved, usage=usage, user_id=user_id)
            context = retrieved["context"] if retrieved["found"] else None
            source_chunks = retrieved["chunks"]
            sources = [{k: c[k] for k in ("source", "page", "chunk_id", "score")}
                       for c in retrieved["chunks"]]
        strategy = routing["execution_strategy"]
        no_generation = conversation["needs_clarification"] or (routing["retrieval_needed"] and not context)
        if not force_model and not no_generation:
            decision = select_best_model(db, routing["complexity"],
                input_tokens=estimate_tokens(build_prompt(resolved, strategy, context, conversation["history"])),
                output_tokens=get_output_token_budget(strategy), policy=policy)
            selected = decision["selected_model"]
        selected = selected or STATIC_MODELS[routing["complexity"]]
        if conversation["needs_clarification"]:
            result = dict(response="Please clarify what you are referring to so I can answer accurately.",
                          model_used="none", latency_ms=0, fallback_used=False, attempts=[], abstained=True)
        else:
            eligible = decision.get("eligible_models") or [m for m in SUITABLE_MODELS[routing["complexity"]]
                                                          if not model_health(db, m)["circuit_open"]]
            # Explicit forced-model experiments may evaluate below the production floor.
            if force_model:
                eligible = [force_model]
            result = execute_query(resolved, selected, routing["strategy"], context,
                retrieval_required=routing["retrieval_needed"], allow_fallback=allow_fallback,
                eligible_models=eligible, history=conversation["history"], source_chunks=source_chunks)
        usage.extend(result["attempts"])
        status = "abstained" if result["abstained"] else "completed"
    except Exception as exc:
        if isinstance(exc, ExecutionError):
            usage.extend(exc.attempts)
        failed = Query(user_id=user_id, session_id=session_id, query_text=clean,
                       intent=routing["intent"], complexity=routing["complexity"],
                       strategy=routing["execution_strategy"], selected_model=selected,
                       model_used="none", status="failed", evaluation_status="skipped",
                       latency_ms=round((time.perf_counter()-start)*1000),
                       routing_details={"error": type(exc).__name__})
        db.add(failed)
        db.flush()
        db.add_all([Usage(query_id=failed.id, **item) for item in usage])
        db.commit()
        raise PipelineUnavailable(failed.id) from exc
    generation = [u for u in usage if u["stage"] == "generation"]
    row = Query(user_id=user_id, session_id=session_id, query_text=clean,
        intent=routing["intent"], complexity=routing["complexity"], strategy=strategy,
        selected_model=selected, model_used=result["model_used"], response=result["response"],
        model_latency_ms=result["latency_ms"], fallback_used=result["fallback_used"],
        status=status, context=context, sources=sources,
        routing_details={"analysis": routing, "decision": decision},
        input_tokens=sum(u["input_tokens"] or 0 for u in generation),
        output_tokens=sum(u["output_tokens"] or 0 for u in generation),
        total_tokens=sum((u["input_tokens"] or 0)+(u["output_tokens"] or 0) for u in generation),
        estimated_cost=sum(u["estimated_cost"] or 0 for u in generation),
        evaluation_status="skipped", evaluation_attempts=0)
    db.add(row)
    db.flush()
    rate = EVALUATION_SAMPLE_RATE if evaluation_rate is None else evaluation_rate
    sample = int(hashlib.sha256(str(row.id).encode()).hexdigest()[:8], 16) / 2**32
    if status == "completed" and sample < rate:
        row.evaluation_status = "pending"
    db.add_all([Usage(query_id=row.id, **item) for item in usage])
    db.add(AuditLog(event_type="request", api_key=str(user_id), detail={
        "query_id": row.id, "status": status, "selected_model": selected,
        "model_used": result["model_used"]}))
    row.latency_ms = round((time.perf_counter()-start)*1000)
    db.commit()
    return row
