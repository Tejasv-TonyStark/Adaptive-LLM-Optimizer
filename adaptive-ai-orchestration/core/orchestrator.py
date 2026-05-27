# core/orchestrator.py

from core.intent_detector import detect_intent
from core.complexity_analyzer import analyze_complexity


def select_strategy(complexity: str) -> str:
    """
    Determines execution strategy based on complexity.
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
    1. Strategy     → based on complexity
    2. RAG needed   → based on intent + specific HR keywords

    Model selection is NOT done here.
    It is handled by the decision engine using probability scores.

    Key fix: intent detector now uses ORG_MARKER + HR_TOPIC logic,
    so "benefits of freelancing" stays general and doesn't
    trigger RAG or wrong model routing.

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
    test_cases = [
        # query                                               expected_complexity  expected_rag
        ("What is Python?",                                   "low",               False),
        ("What are the benefits of being a freelancer?",      "low",               False),
        ("What are some salary negotiation tips?",            "low",               False),
        ("Is this related to the RAG documents I uploaded?",  "low",               False),
        ("Explain how neural networks work",                  "medium",            False),
        ("Compare Python vs Java",                            "medium",            False),
        ("What is our leave policy?",                         "low",               True),
        ("What does our contract say about resignation?",     "low",               True),
        ("Summarize our employee handbook",                   "medium",            True),
        ("Compare transformers vs RNN in detail",             "high",              False),
        ("Find contradictions across our contracts",          "high",              True),
    ]

    print("\n── Orchestration Results ──\n")

    passed = 0
    failed = 0

    for query, exp_complexity, exp_rag in test_cases:
        result = orchestrate(query)

        complexity_ok = result["complexity"]       == exp_complexity
        rag_ok        = result["retrieval_needed"] == exp_rag
        overall       = complexity_ok and rag_ok

        icon = "✅" if overall else "❌"
        if overall: passed += 1
        else:        failed += 1

        print(f"{icon} {query}")
        print(f"   Intent     : {result['intent']} (conf={result['confidence']})")
        print(f"   Complexity : expected={exp_complexity:6}  got={result['complexity']:6}  {'✓' if complexity_ok else '✗'}")
        print(f"   RAG        : expected={str(exp_rag):5}   got={str(result['retrieval_needed']):5}  {'✓' if rag_ok else '✗'}")
        print(f"   Strategy   : {result['execution_strategy']}")
        print()

    print(f"── Results: {passed} passed / {failed} failed ──")
