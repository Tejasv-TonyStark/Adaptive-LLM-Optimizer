# core/complexity_analyzer.py

# ──────────────────────────────────────────
# DOMAIN OVERRIDE — ALWAYS HIGH
# ──────────────────────────────────────────

HIGH_DOMAIN_OVERRIDES = [
    # Mathematical / formal reasoning
    "mathematical",
    "mathematics",
    "theorem",
    "proof",
    "derivation",
    "derive",
    "calculus",
    "algebraic",
    "statistical significance",
    "probability distribution",
    "eigenvalue",
    "gradient descent",
    "backpropagation",
    "foundations of",
    "formal definition",

    # Legal / document reasoning
    "contradictions",
    "contradiction",
    "across contracts",
    "across multiple",
    "legal implications",
    "compliance risk",
    "liability",
    "jurisdiction",
    "discrepancy",
    "discrepancies",
    "cross-reference",
    "clause",
    "clauses",

    # Deep analysis
    "critically analyze",
    "critically evaluate",
    "root cause analysis",
    "systemic failure",
    "architectural implications",
    "failure modes",
    "reconcile",
    "inconsistencies",
    "multi-document",
    "synthesize across",

    # architecture comparisons
    "compare architectures",
    "architectures in detail",
]


# ──────────────────────────────────────────
# MEDIUM FLOOR
# ──────────────────────────────────────────

MEDIUM_FLOOR_KEYWORDS = [
    "summarize",
    "summarise",
    "summary",
    "give me an overview",
    "overview of",
    "brief me on",
    "recap",

    "explain how",
    "explain what",
    "explain why",
    "explain the concept",
    "explain the difference",
    "how does",
    "how do",

    "compare",
    "contrast",
    "difference between",
    "vs",
    "versus",
    "pros and cons",
    "advantages and disadvantages",
    "similarities and differences",
]


# ──────────────────────────────────────────
# WEIGHTED SCORING
# ──────────────────────────────────────────

SCORING_KEYWORDS = {
    "analyze": 3,
    "analyse": 3,
    "evaluate": 3,
    "critique": 3,
    "differentiate": 3,
    "distinguish": 3,
    "assess": 3,
    "justify": 3,

    "elaborate": 2,
    "discuss": 2,
    "examine": 2,
    "investigate": 2,
    "identify": 2,
    "determine": 2,

    "describe": 1,
    "outline": 1,
    "list": 1,
}


# ──────────────────────────────────────────
# RAG DETECTION
# ──────────────────────────────────────────

RAG_KEYWORDS = [
    "policy",
    "policies",
    "clause",
    "clauses",
    "section",
    "guideline",
    "reimbursement",
    "leave",
    "salary",
    "benefits",
    "allowance",
    "resignation",
    "notice period",
    "contract",
    "agreement",
    "rules",
    "regulation",
    "procedure",
    "handbook",
    "document",
    "according to",
    "as per",
    "what does it say",
    "our company",
    "the company",
]


LOW_COMPLEXITY_STARTERS = [
    "what is",
    "who is",
    "define",
    "what are",
    "when is",
    "where is",
    "how many",
    "what does",
    "name the",
    "list the",
    "give me the",
    "tell me the",
]


# ──────────────────────────────────────────
# MAIN ANALYZER
# ──────────────────────────────────────────

def analyze_complexity(query: str, intent: str) -> dict:
    query_lower = query.lower()
    words = query_lower.split()
    word_count = len(words)

    rag = _check_rag(query_lower, intent)

    # HIGH overrides
    for keyword in HIGH_DOMAIN_OVERRIDES:
        if keyword in query_lower:
            return _result(
                complexity="high",
                score=99,
                retrieval_needed=rag,
                trigger=f"override:{keyword}"
            )

    # compare + architecture
    if "compare" in query_lower and "architecture" in query_lower:
        return _result(
            complexity="high",
            score=99,
            retrieval_needed=rag,
            trigger="override:compare+architecture"
        )

    if "compare" in query_lower and "in detail" in query_lower:
        return _result(
            complexity="high",
            score=99,
            retrieval_needed=rag,
            trigger="override:compare+detail"
        )

    # medium floor
    medium_triggered = False

    for keyword in MEDIUM_FLOOR_KEYWORDS:
        if keyword in query_lower:
            medium_triggered = True
            break

    if intent == "specific" and any(
        kw in query_lower
        for kw in ["summarize", "summarise", "overview", "recap"]
    ):
        medium_triggered = True

    # scoring
    score = 0

    # query length
    if word_count < 6:
        score += 0
    elif word_count <= 10:
        score += 1
    elif word_count <= 18:
        score += 2
    else:
        score += 3

    # weighted keywords
    for keyword, weight in SCORING_KEYWORDS.items():
        if keyword in query_lower:
            score += weight

    # simple starter penalty
    for starter in LOW_COMPLEXITY_STARTERS:
        if query_lower.startswith(starter):
            score -= 2
            break

    # multipart reasoning bonus
    multi_part = [
        "and also",
        "as well as",
        "furthermore",
        "in addition",
        "moreover",
        "while also",
        "additionally",
        "at the same time",
    ]

    if any(term in query_lower for term in multi_part):
        score += 2

    score = max(score, 0)

    # raw classification
    if score <= 1:
        raw = "low"
    elif score <= 4:
        raw = "medium"
    else:
        raw = "high"

    # enforce medium floor
    if medium_triggered and raw == "low":
        final = "medium"
        trigger = "medium_floor"
    else:
        final = raw
        trigger = None

    return _result(
        complexity=final,
        score=score,
        retrieval_needed=rag,
        trigger=trigger
    )


# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────

def _check_rag(query_lower: str, intent: str) -> bool:
    if intent == "general":
        return False

    return any(keyword in query_lower for keyword in RAG_KEYWORDS)


def _result(complexity, score, retrieval_needed, trigger):
    return {
        "complexity": complexity,
        "score": score,
        "retrieval_needed": retrieval_needed,
        "trigger": trigger,
    }


# ──────────────────────────────────────────
# TEST RUNNER
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("What is Python?", "general", "low", False),
        ("Who is Alan Turing?", "general", "low", False),
        ("What is our leave policy?", "specific", "low", True),
        ("Explain how neural networks work", "general", "medium", False),
        ("Summarize our leave policy", "specific", "medium", True),
        ("Compare Python vs Java", "general", "medium", False),
        ("Explain the mathematical foundations of backpropagation", "general", "high", False),
        ("Compare transformers vs RNN architectures in detail", "general", "high", False),
        ("Compare clauses across contracts and find contradictions", "specific", "high", True),
        ("Critically analyze the implications of our resignation clause", "specific", "high", True),
        ("Synthesize findings across multiple policy documents", "specific", "high", True),
    ]

    passed = 0
    failed = 0

    print("\n── Complexity Classifier Results ──\n")

    for query, intent, expected_complexity, expected_rag in test_cases:
        result = analyze_complexity(query, intent)

        complexity_ok = result["complexity"] == expected_complexity
        rag_ok = result["retrieval_needed"] == expected_rag

        overall = complexity_ok and rag_ok

        if overall:
            passed += 1
            icon = "✅"
        else:
            failed += 1
            icon = "❌"

        print(f"{icon} {query}")
        print(f"   Complexity: expected={expected_complexity:6} got={result['complexity']:6}")
        print(f"   RAG:        expected={expected_rag} got={result['retrieval_needed']}")

        if result["trigger"]:
            print(f"   Trigger:    {result['trigger']}")

        print(f"   Score:      {result['score']}")
        print()

    print(f"── Results: {passed} passed / {failed} failed ──")