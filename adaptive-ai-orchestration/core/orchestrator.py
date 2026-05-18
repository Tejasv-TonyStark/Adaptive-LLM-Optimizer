# core/orchestrator.py

from core.intent_detector import detect_intent
from core.complexity_analyzer import analyze_complexity


def select_model(complexity: str) -> tuple[str, str]:
    """
    Decision 1 — Select model based on complexity ONLY.
    RAG is completely irrelevant to this decision.

    Returns:
        tuple of (model, strategy)
    """
    if complexity == "low":
        return "nova-micro", "fast"
    elif complexity == "medium":
        return "llama3-8b", "reasoning"
    else:
        return "haiku", "reasoning"


def decide_retrieval(intent: str, retrieval_needed: bool) -> bool:
    """
    Decision 2 — Decide if RAG retrieval is needed.
    Model selection is completely irrelevant to this decision.

    Returns:
        bool — whether to retrieve documents
    """
    if intent == "general":
        return False
    return retrieval_needed


def orchestrate(query: str) -> dict:
    """
    Master routing function.

    Makes TWO completely independent decisions:

    Decision 1 → Which model?
                 Based on: complexity, cost, latency
                 Nova Micro  → low complexity
                 Llama 3.1   → medium complexity
                 Claude Haiku → high complexity

    Decision 2 → RAG needed?
                 Based on: intent, keyword matching
                 True  → retrieve document chunks first
                 False → answer directly from model knowledge

    Valid combinations:
        Nova  + no RAG  → simple general question
        Nova  + RAG     → simple document lookup
        Llama + no RAG  → medium reasoning question
        Llama + RAG     → medium document reasoning
        Haiku + no RAG  → complex reasoning question
        Haiku + RAG     → complex document reasoning
    """

    # ── Step 1: Detect intent ──
    intent_result = detect_intent(query)
    intent        = intent_result["intent"]
    confidence    = intent_result["confidence"]

    # ── Step 2: Analyze complexity ──
    complexity_result    = analyze_complexity(query, intent)
    complexity           = complexity_result["complexity"]
    raw_retrieval_needed = complexity_result["retrieval_needed"]

    # ── Decision 1: Model selection (complexity only) ──
    model, strategy = select_model(complexity)

    # ── Decision 2: RAG decision (intent + keywords only) ──
    retrieval_needed = decide_retrieval(intent, raw_retrieval_needed)

    # ── Step 3: Adjust strategy label if RAG needed ──
    # Strategy label reflects execution path, not model change
    if retrieval_needed:
        execution_strategy = f"{strategy}+rag"
    else:
        execution_strategy = strategy

    return {
        "intent":             intent,
        "confidence":         confidence,
        "complexity":         complexity,
        "model":              model,
        "strategy":           strategy,
        "retrieval_needed":   retrieval_needed,
        "execution_strategy": execution_strategy
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        # Simple general
        ("What is Python?",                                           "Nova + no RAG expected"),
        # Simple document lookup
        ("What is our leave policy?",                                 "Nova + RAG expected"),
        # Medium reasoning
        ("Compare transformers vs RNN architectures in detail",       "Llama + no RAG expected"),
        # Medium document reasoning
        ("Summarize our leave policy and compare annual vs maternity","Llama + RAG expected"),
        # Complex reasoning
        ("Explain the mathematical foundations of backpropagation",   "Haiku + no RAG expected"),
        # Complex document reasoning
        ("Compare clauses across contracts and find contradictions",  "Haiku + RAG expected"),
    ]

    print("\n── Orchestration Results ──\n")
    for query, expected in test_queries:
        result = orchestrate(query)
        print(f"Query:              {query}")
        print(f"Expected:           {expected}")
        print(f"Intent:             {result['intent']}")
        print(f"Complexity:         {result['complexity']}")
        print(f"Model:              {result['model']}")
        print(f"Retrieval needed:   {result['retrieval_needed']}")
        print(f"Execution strategy: {result['execution_strategy']}")
        print()