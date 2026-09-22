import time
from Execution.Bedrock_client import invoke_model
from Execution.prompt_builder import build_prompt
from tracking.usage import usage_record
from RAG.evidence import render_evidence, extractive_fallback, ABSTENTION
FALLBACK_CHAIN = {
    "llama3-70b": ["llama3-8b", "nova-micro"],
    "llama3-8b": ["nova-micro", "llama3-70b"],
    "nova-micro": ["llama3-8b", "llama3-70b"],
}
OUTPUT_TOKEN_BUDGETS = {"fast": 160, "reasoning": 256, "rag": 512, "default": 256}
MAX_GENERAL_RESPONSE_CHARS = 600
class ExecutionError(RuntimeError):
    def __init__(self, attempts):
        super().__init__("All configured models failed.")
        self.attempts = attempts
def get_output_token_budget(strategy):
    return OUTPUT_TOKEN_BUDGETS.get(strategy, OUTPUT_TOKEN_BUDGETS["default"])

def limit_general_response(text):
    """Enforce a UI-safe answer size even if a provider ignores the prompt."""
    text = text.strip()
    if len(text) <= MAX_GENERAL_RESPONSE_CHARS:
        return text
    shortened = text[:MAX_GENERAL_RESPONSE_CHARS - 1].rsplit(None, 1)[0].rstrip()
    return (shortened or text[:MAX_GENERAL_RESPONSE_CHARS - 1].rstrip()) + "…"
def execute_query(query, model, strategy, context=None, retrieval_required=False, allow_fallback=True,
                  eligible_models=None, history=None, source_chunks=None):
    if model not in FALLBACK_CHAIN:
        raise ValueError("Unknown model")
    required = retrieval_required or strategy == "rag"
    if required and not context:
        return dict(response=ABSTENTION, model_used="none", selected_model=model,
                    latency_ms=0, fallback_used=False, input_tokens=0, output_tokens=0,
                    error=None, attempts=[], abstained=True)
    prompt_strategy = "rag" if required or context else strategy
    prompt = build_prompt(query, prompt_strategy, context, history=history)
    attempts = []
    models = [model, *(FALLBACK_CHAIN[model] if allow_fallback else [])]
    if eligible_models is not None:
        models = [m for m in models if m in eligible_models]
    invalid_evidence = False
    for candidate in models:
        started = time.perf_counter()
        try:
            result = invoke_model(candidate, prompt,
                                  max_output_tokens=get_output_token_budget(prompt_strategy))
            elapsed = round((time.perf_counter()-started)*1000)
            record = usage_record(candidate, "generation", prompt, result, elapsed)
            attempts.append(record)
            if not result["text"].strip():
                record["status"] = "failed"
                record["error"] = "EmptyResponse"
                continue
            if result.get("stop_reason") in {"length", "max_tokens", "max_token", "maxTokens"}:
                record.update(status="failed", error="TruncatedResponse")
                continue
            response, abstained = result["text"], False
            if required:
                try:
                    response, abstained = render_evidence(response, context, source_chunks, query=query)
                except (ValueError, TypeError):
                    record.update(status="failed", error="InvalidEvidence")
                    invalid_evidence = True
                    continue
            else:
                response = limit_general_response(response)
            return dict(response=response, model_used=candidate, selected_model=model,
                        latency_ms=elapsed, fallback_used=candidate != model,
                        input_tokens=record["input_tokens"], output_tokens=record["output_tokens"],
                        attempts=attempts, abstained=abstained, error=None)
        except Exception as exc:
            attempts.append(usage_record(candidate, "generation", prompt, {},
                round((time.perf_counter()-started)*1000), error=exc))
    if invalid_evidence:
        response = extractive_fallback(query, source_chunks)
        if response:
            return dict(response=response, model_used="extractive-rag", selected_model=model,
                        latency_ms=0, fallback_used=True, input_tokens=0, output_tokens=0,
                        error="InvalidEvidence", attempts=attempts, abstained=False)
        return dict(response="I could not verify supporting passages in the available documents.",
                    model_used="none", selected_model=model, latency_ms=0,
                    fallback_used=len(attempts)>1, input_tokens=0, output_tokens=0,
                    error="InvalidEvidence", attempts=attempts, abstained=True)
    raise ExecutionError(attempts)
