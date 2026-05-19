# core/orchestrator.py

from core.intent_detector import detect_intent
from core.complexity_analyzer import analyze_complexity


def select_strategy(complexity: str) -> str:
    """
    Determines execution strategy based on complexity only.
    Model selection is handled by the decision engine separately.
    """
    if complexity == "low":
        return "fast"
    elif complexity == "medium":
        return "reasoning"
    else:
        return "reasoning"


def orchestrate(query: str) -> dict:
    """
    Master routing function.

    Makes TWO independent decisions:
    1. Strategy → based on complexity
    2. RAG needed → based on intent + keywords

    Model selection is NOT done here.
    It is handled by the decision engine using probability scores.

    Returns:
        dict with intent, complexity, strategy,
        retrieval_needed, execution_strategy
    """

    # ── Step 1: Detect intent ──
    intent_result = detect_intent(query)
    intent        = intent_result["intent"]
    confidence    = intent_result["confidence"]

    # ── Step 2: Analyze complexity ──
    complexity_result = analyze_complexity(query, intent)
    complexity        = complexity_result["complexity"]
    retrieval_needed  = complexity_result["retrieval_needed"]

    # ── Step 3: Determine strategy ──
    strategy = select_strategy(complexity)

    # ── Step 4: Build execution strategy label ──
    if retrieval_needed:
        execution_strategy = f"{strategy}+rag"
    else:
        execution_strategy = strategy

    return {
        "intent":             intent,
        "confidence":         confidence,
        "complexity":         complexity,
        "retrieval_needed":   retrieval_needed,
        "strategy":           strategy,
        "execution_strategy": execution_strategy
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is Python?",
        "Compare transformers vs RNN architectures in detail",
        "What is our leave policy?",
        "What does our contract say about resignation?",
        "Explain how neural networks work",
        "What are our salary benefits?"
    ]

    print("\n── Orchestration Results ──\n")
    for query in test_queries:
        result = orchestrate(query)
        print(f"Query:              {query}")
        print(f"Intent:             {result['intent']}")
        print(f"Complexity:         {result['complexity']}")
        print(f"Retrieval needed:   {result['retrieval_needed']}")
        print(f"Strategy:           {result['strategy']}")
        print(f"Execution strategy: {result['execution_strategy']}")
        print()