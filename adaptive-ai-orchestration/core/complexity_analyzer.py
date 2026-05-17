# core/complexity_analyzer.py

# ──────────────────────────────────────────
# KEYWORDS
# ──────────────────────────────────────────

REASONING_KEYWORDS = [
    "compare", "contrast", "explain", "analyze",
    "difference", "vs", "versus", "why", "how does",
    "evaluate", "discuss", "elaborate", "distinguish",
    "pros and cons", "advantages", "disadvantages",
    "impact", "effect", "relationship", "correlation"
]

RAG_KEYWORDS = [
    "policy", "policies", "clause", "section",
    "guideline", "reimbursement", "leave", "salary",
    "benefits", "allowance", "resignation", "notice period",
    "contract", "agreement", "rules", "regulation",
    "procedure", "process", "handbook", "document"
]

LOW_COMPLEXITY_STARTERS = [
    "what is", "who is", "define", "what are",
    "when is", "where is", "how many", "what does"
]

HIGH_COMPLEXITY_STARTERS = [
    "compare", "explain", "analyze", "evaluate",
    "discuss", "elaborate", "why does", "how does"
]


# ──────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────

def analyze_complexity(query: str, intent: str) -> dict:
    """
    Analyzes the complexity of a query using rule based scoring.
    Also checks if RAG retrieval is needed.

    Args:
        query:  the user question
        intent: output from intent_detector (general/specific/unknown)

    Returns:
        dict with complexity (low/medium/high) and retrieval_needed flag
    """
    query_lower = query.lower()
    words       = query_lower.split()
    word_count  = len(words)
    score       = 0

    # ── Factor 1: Query length ──
    if word_count < 8:
        score += 1
    elif word_count <= 15:
        score += 2
    else:
        score += 3

    # ── Factor 2: Reasoning keywords ──
    reasoning_hits = sum(1 for kw in REASONING_KEYWORDS if kw in query_lower)
    score += reasoning_hits * 2

    # ── Factor 3: Question structure bias ──
    for starter in LOW_COMPLEXITY_STARTERS:
        if query_lower.startswith(starter):
            score -= 1
            break

    for starter in HIGH_COMPLEXITY_STARTERS:
        if query_lower.startswith(starter):
            score += 2
            break

    # ── Final complexity classification ──
    if score <= 3:
        complexity = "low"
    elif score <= 7:
        complexity = "medium"
    else:
        complexity = "high"

    # ── RAG check ──
    # If intent is general → never use RAG
    # If intent is specific/unknown → check keywords
    if intent == "general":
        retrieval_needed = False
    else:
        rag_hits         = sum(1 for kw in RAG_KEYWORDS if kw in query_lower)
        retrieval_needed = rag_hits > 0

    return {
        "complexity":        complexity,
        "score":             score,
        "retrieval_needed":  retrieval_needed
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("What is Python?",                                      "general"),
        ("Explain how neural networks work",                     "general"),
        ("Compare transformers vs RNN for NLP tasks in detail",  "general"),
        ("What is our leave policy?",                            "specific"),
        ("What does our contract say about resignation?",        "specific"),
        ("Compare our salary benefits vs industry standards",    "specific"),
    ]

    print("\n── Complexity Analysis Results ──\n")
    for query, intent in test_cases:
        result = analyze_complexity(query, intent)
        print(f"Query:            {query}")
        print(f"Intent:           {intent}")
        print(f"Complexity:       {result['complexity']}")
        print(f"Score:            {result['score']}")
        print(f"Retrieval needed: {result['retrieval_needed']}")
        print()