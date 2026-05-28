# Execution/Execution_layer.py

import time
from Execution.Bedrock_client import invoke_model
from Execution.prompt_builder import build_prompt


# ──────────────────────────────────────────
# FALLBACK CHAIN
# Primary fails → try these in order
# ──────────────────────────────────────────

FALLBACK_CHAIN = {
    "haiku":      ["llama3-8b", "nova-micro"],
    "llama3-8b":  ["nova-micro", "haiku"],
    "nova-micro": ["llama3-8b",  "haiku"],
}


def execute_query(query: str, model: str, strategy: str,
                  context: str = None) -> dict:
    """
    Executes query against selected model with fallback chain.

    Returns dict:
        response       — model's answer text
        model_used     — which model actually answered
        latency_ms     — wall-clock ms for the call
        fallback_used  — True if primary model failed
        input_tokens   — prompt token count (from Bedrock metadata, or None)
        output_tokens  — response token count (from Bedrock metadata, or None)
        error          — error message if all models failed, else None
    """
    prompt     = build_prompt(query, strategy, context)
    start_time = time.time()

    # ── Primary attempt ────────────────────
    try:
        result     = invoke_model(model, prompt)   # returns {text, input_tokens, output_tokens}
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "response":      result["text"],
            "model_used":    model,
            "latency_ms":    latency_ms,
            "fallback_used": False,
            "input_tokens":  result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "error":         None,
        }

    except Exception as primary_error:
        print(f"⚠️  Primary model {model} failed: {primary_error}")

    # ── Fallback chain ─────────────────────
    for fallback_model in FALLBACK_CHAIN.get(model, []):
        try:
            print(f"Trying fallback → {fallback_model}")

            start_time = time.time()
            result     = invoke_model(fallback_model, prompt)
            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "response":      result["text"],
                "model_used":    fallback_model,
                "latency_ms":    latency_ms,
                "fallback_used": True,
                "input_tokens":  result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "error":         None,
            }

        except Exception as fallback_error:
            print(f"Fallback {fallback_model} failed: {fallback_error}")

    # ── All models failed ──────────────────
    return {
        "response":      "All models failed. Please try again later.",
        "model_used":    model,
        "latency_ms":    int((time.time() - start_time) * 1000),
        "fallback_used": True,
        "input_tokens":  None,
        "output_tokens": None,
        "error":         str(primary_error),
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Execution Layer Test ──\n")

    test_cases = [
        {"query": "What is Python?",                            "model": "nova-micro", "strategy": "fast",      "context": None},
        {"query": "Compare transformers vs RNN architectures",  "model": "llama3-8b",  "strategy": "reasoning", "context": None},
        {"query": "What is our leave policy?",                  "model": "nova-micro", "strategy": "rag",
         "context": "Employees are entitled to 18 days of annual leave per year."},
    ]

    for test in test_cases:
        print(f"Query:    {test['query']}")
        print(f"Model:    {test['model']}  |  Strategy: {test['strategy']}")

        result = execute_query(**test)

        print(f"Response: {result['response'][:150]}...")
        print(f"Tokens:   input={result['input_tokens']}  output={result['output_tokens']}")
        print(f"Latency:  {result['latency_ms']} ms  |  Fallback: {result['fallback_used']}")
        print("-" * 50)

    print("✅ Execution layer working correctly!")
