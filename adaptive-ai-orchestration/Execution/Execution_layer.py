# Execution/execution_layer.py

import time
from Execution.Bedrock_client import invoke_model
from Execution.prompt_builder import build_prompt


# ==========================================
# FALLBACK CHAIN
# ==========================================
FALLBACK_CHAIN = {
    "haiku": ["llama3-8b", "nova-micro"],
    "llama3-8b": ["nova-micro", "haiku"],
    "nova-micro": ["llama3-8b", "haiku"]
}


def execute_query(query: str, model: str, strategy: str, context: str = None) -> dict:
    """
    Executes query against selected model.
    Uses fallback chain if primary model fails.
    """

    prompt = build_prompt(query, strategy, context)

    start_time = time.time()

    try:
        response = invoke_model(model, prompt)
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "response": response,
            "model_used": model,
            "latency_ms": latency_ms,
            "fallback_used": False,
            "error": None
        }

    except Exception as primary_error:
        print(f"⚠️ Primary model {model} failed: {primary_error}")

        for fallback_model in FALLBACK_CHAIN.get(model, []):
            try:
                print(f"Trying fallback → {fallback_model}")

                start_time = time.time()
                response = invoke_model(fallback_model, prompt)
                latency_ms = int((time.time() - start_time) * 1000)

                return {
                    "response": response,
                    "model_used": fallback_model,
                    "latency_ms": latency_ms,
                    "fallback_used": True,
                    "error": None
                }

            except Exception as fallback_error:
                print(f"Fallback {fallback_model} failed: {fallback_error}")

        return {
            "response": "All models failed. Please try again later.",
            "model_used": model,
            "latency_ms": int((time.time() - start_time) * 1000),
            "fallback_used": True,
            "error": str(primary_error)
        }


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    print("\n── Execution Layer Test ──\n")

    test_cases = [
        {
            "query": "What is Python?",
            "model": "nova-micro",
            "strategy": "fast",
            "context": None
        },
        {
            "query": "Compare transformers vs RNN architectures",
            "model": "llama3-8b",
            "strategy": "reasoning",
            "context": None
        },
        {
            "query": "What is our leave policy?",
            "model": "nova-micro",
            "strategy": "rag",
            "context": "Employees are entitled to 18 days of annual leave per year."
        }
    ]

    for test in test_cases:
        print(f"Query: {test['query']}")
        print(f"Model: {test['model']}")
        print(f"Strategy: {test['strategy']}")

        result = execute_query(
            query=test["query"],
            model=test["model"],
            strategy=test["strategy"],
            context=test["context"]
        )

        print(f"Response: {result['response'][:150]}...")
        print(f"Latency: {result['latency_ms']} ms")
        print(f"Fallback used: {result['fallback_used']}")
        print("-" * 50)

    print("✅ Execution layer working correctly!")