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

    # Architecture comparisons
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
# Specific HR/company keywords only.
# Generic words like "benefits", "rules",
# "document" removed to avoid false triggers.
# ──────────────────────────────────────────

RAG_KEYWORDS = [
    # HR specific
    "leave policy",
    "notice period",
    "resignation",
    "reimbursement",
    "allowance",
    "salary structure",
    "salary details",
    "appraisal",
    "performance review",
    "onboarding",
    "offboarding",
    "probation",

    # Document references
    "handbook",
    "employee handbook",
    "hr policy",
    "company policy",
    "company guidelines",
    "our policy",
    "our guidelines",

    # Legal/contract
    "clause",
    "clauses",
    "contract terms",
    "agreement terms",
    "regulation",
    "compliance",

    # Explicit references
    "according to",
    "as per",
    "as mentioned in",
    "what does it say",
    "what does the document",
    "our company",
    "the company says",
    "info services",
    "infoservices",
]


# ──────────────────────────────────────────
# LOW COMPLEXITY STARTERS
# ──────────────────────────────────────────

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
    "is this",
    "are there",
    "can you tell",
    "do you know",
]


# ──────────────────────────────────────────
# MAIN ANALYZER
# ──────────────────────────────────────────

def analyze_complexity(query: str, intent: str) -> dict:
    query_lower = query.lower()
    words       = query_lower.split()
    word_count  = len(words)

    rag = _check_rag(query_lower, intent)

    # ── HIGH overrides ──
    for keyword in HIGH_DOMAIN_OVERRIDES:
        if keyword in query_lower:
            return _result(
                complexity       = "high",
                score            = 99,
                retrieval_needed = rag,
                trigger          = f"override:{keyword}"
            )

    # compare + architecture special case
    if "compare" in query_lower and "architecture" in query_lower:
        return _result(
            complexity       = "high",
            score            = 99,
            retrieval_needed = rag,
            trigger          = "override:compare+architecture"
        )

    if "compare" in query_lower and "in detail" in query_lower:
        return _result(
            complexity       = "high",
            score            = 99,
            retrieval_needed = rag,
            trigger          = "override:compare+detail"
        )

    # ── Medium floor ──
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

    # ── Scoring ──
    score = 0

    # Query length contribution
    if word_count < 6:
        score += 0
    elif word_count <= 10:
        score += 1
    elif word_count <= 18:
        score += 2
    else:
        score += 3

    # Weighted keywords
    for keyword, weight in SCORING_KEYWORDS.items():
        if keyword in query_lower:
            score += weight

    # Simple starter penalty — strongly penalise obvious simple questions
    for starter in LOW_COMPLEXITY_STARTERS:
        if query_lower.startswith(starter):
            score -= 3   # increased from -2 to -3
            break

    # Multi-part reasoning bonus
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

    # ── Raw classification ──
    if score <= 1:
        raw = "low"
    elif score <= 4:
        raw = "medium"
    else:
        raw = "high"

    # ── Enforce medium floor ──
    if medium_triggered and raw == "low":
        final   = "medium"
        trigger = "medium_floor"
    else:
        final   = raw
        trigger = None

    return _result(
        complexity       = final,
        score            = score,
        retrieval_needed = rag,
        trigger          = trigger
    )


# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────

def _check_rag(query_lower: str, intent: str) -> bool:
    """
    Only trigger RAG if query contains specific HR/company phrases.
    Generic intent alone is never enough.
    """
    return any(keyword in query_lower for keyword in RAG_KEYWORDS)


def _result(complexity, score, retrieval_needed, trigger):
    return {
        "complexity":       complexity,
        "score":            score,
        "retrieval_needed": retrieval_needed,
        "trigger":          trigger,
    }


# ──────────────────────────────────────────
# TEST RUNNER
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # query                                                  intent      expected_complexity  expected_rag
        ("What is Python?",                                      "general",  "low",               False),
        ("Who is Alan Turing?",                                  "general",  "low",               False),
        ("What are the benefits of being a freelancer?",         "general",  "low",               False),
        ("Is this related to RAG documents?",                    "general",  "low",               False),
        ("What is our leave policy?",                            "specific", "low",               True),
        ("What is the notice period for resignation?",           "specific", "low",               True),
        ("Explain how neural networks work",                     "general",  "medium",            False),
        ("Summarize our employee handbook",                      "specific", "medium",            True),
        ("Compare Python vs Java",                               "general",  "medium",            False),
        ("How does backpropagation work?",                       "general",  "medium",            False),
        ("Explain the mathematical foundations of backprop",     "general",  "high",              False),
        ("Compare transformers vs RNN architectures in detail",  "general",  "high",              False),
        ("Find contradictions across contracts",                 "specific", "high",              True),
        ("Critically analyze the resignation clause",            "specific", "high",              True),
    ]

    passed = 0
    failed = 0

    print("\n── Complexity Classifier Results ──\n")

    for query, intent, expected_complexity, expected_rag in test_cases:
        result = analyze_complexity(query, intent)

        complexity_ok = result["complexity"] == expected_complexity
        rag_ok        = result["retrieval_needed"] == expected_rag
        overall       = complexity_ok and rag_ok

        if overall:
            passed += 1
            icon = "✅"
        else:
            failed += 1
            icon = "❌"

        print(f"{icon} {query}")
        print(f"   Complexity : expected={expected_complexity:6}  got={result['complexity']:6}  {'✓' if complexity_ok else '✗'}")
        print(f"   RAG        : expected={str(expected_rag):5}   got={str(result['retrieval_needed']):5}  {'✓' if rag_ok else '✗'}")

        if result["trigger"]:
            print(f"   Trigger    : {result['trigger']}")

        print(f"   Score      : {result['score']}")
        print()

    print(f"── Results: {passed} passed / {failed} failed ──")
