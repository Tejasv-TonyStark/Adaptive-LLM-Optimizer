# core/orchestrator.py

from core.intent_detector import detect_intent
from core.complexity_analyzer import analyze_complexity


def orchestrate(query: str) -> dict:
    """
    Master function that runs both detectors
    and returns a combined routing decision.

    This is what main.py will call for every request.

    Returns:
        dict with intent, complexity, retrieval_needed
    """

    # Step 1 — Detect intent
    intent_result = detect_intent(query)
    intent        = intent_result["intent"]
    confidence    = intent_result["confidence"]

    # Step 2 — Analyze complexity
    complexity_result = analyze_complexity(query, intent)
    complexity        = complexity_result["complexity"]
    retrieval_needed  = complexity_result["retrieval_needed"]

    # Step 3 — Determine strategy
    if retrieval_needed:
        strategy = "rag"
    elif complexity == "high":
        strategy = "reasoning"
    else:
        strategy = "fast"

    return {
        "intent":           intent,
        "confidence":       confidence,
        "complexity":       complexity,
        "retrieval_needed": retrieval_needed,
        "strategy":         strategy
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is Python?",
        "Compare transformers vs RNN in detail",
        "What is our leave policy?",
        "What does our contract say about resignation?",
        "Explain how neural networks work",
        "What are our salary benefits?"
    ]

    print("\n── Orchestration Results ──\n")
    for query in test_queries:
        result = orchestrate(query)
        print(f"Query:            {query}")
        print(f"Intent:           {result['intent']}")
        print(f"Complexity:       {result['complexity']}")
        print(f"Retrieval needed: {result['retrieval_needed']}")
        print(f"Strategy:         {result['strategy']}")
        print()