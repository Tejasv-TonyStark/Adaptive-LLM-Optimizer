# Evaluation/Evaluator.py

import json
from Execution.Bedrock_client import invoke_model

# ──────────────────────────────────────────
# JUDGE PROMPT — improved with hallucination check
# Forces the judge to verify claims against context
# ──────────────────────────────────────────

JUDGE_PROMPT = """You are a strict AI response evaluator. Your job is to score an AI answer.

QUESTION:
{query}

AI ANSWER:
{response}

{context_section}

SCORING RULES:
- relevance (0.0-1.0): Does the answer directly address the question?
  1.0 = fully answers the question
  0.5 = partially answers
  0.0 = off-topic or ignores the question

- correctness (0.0-1.0): Are all claims in the answer accurate?
  With context: are all claims supported by the context? (faithfulness)
  Without context: is the answer factually accurate based on general knowledge?
  1.0 = every claim is accurate/supported
  0.5 = mostly accurate, minor issues
  0.0 = contains wrong or contradicting information

- completeness (0.0-1.0): Does the answer cover everything important?
  1.0 = nothing important missing
  0.5 = some relevant points omitted
  0.0 = major gaps or very incomplete

- hallucination_flags: List specific claims in the answer that are NOT
  supported by the context (if context was provided). Write [] if none
  or if no context was provided.

Return ONLY this JSON, nothing else, no markdown, no explanation outside the JSON:
{{"relevance": 0.0, "correctness": 0.0, "completeness": 0.0, "hallucination_flags": [], "reasoning": "one sentence"}}"""

CONTEXT_SECTION = """DOCUMENT CONTEXT (ground truth — answer must be faithful to this):
{context}

IMPORTANT: If the answer makes claims NOT found in the context above,
those are hallucinations. Mark them in hallucination_flags."""

NO_CONTEXT_NOTE = """NOTE: No document context was provided.
Score correctness based on general factual accuracy."""


# ──────────────────────────────────────────
# SCORING FORMULA
# ──────────────────────────────────────────

def calculate_quality_score(relevance: float,
                             correctness: float,
                             completeness: float,
                             hallucination_count: int = 0) -> float:
    """
    quality = 0.40*correctness + 0.35*relevance + 0.25*completeness
    Penalised by 0.05 per hallucination flag, capped at -0.20 total.
    """
    base = (
        (0.40 * correctness)  +
        (0.35 * relevance)    +
        (0.25 * completeness)
    )
    penalty = min(hallucination_count * 0.05, 0.20)
    return round(max(base - penalty, 0.0), 4)


# ──────────────────────────────────────────
# RETRIEVAL QUALITY CHECK
# Runs before the judge — checks if retrieved
# chunks are actually relevant to the query.
# Returns a score 0.0-1.0 and a warning flag.
# ──────────────────────────────────────────

def check_retrieval_quality(query: str, context: str) -> dict:
    """
    Basic heuristic check — does the context contain
    keywords from the query? If very low overlap,
    the retrieval may have fetched wrong chunks.
    """
    if not context:
        return {"score": 1.0, "warning": False}

    query_words = set(
        w.lower() for w in query.split()
        if len(w) > 3
    )
    context_lower = context.lower()

    if not query_words:
        return {"score": 1.0, "warning": False}

    hits  = sum(1 for w in query_words if w in context_lower)
    score = hits / len(query_words)

    return {
        "score":   round(score, 2),
        "warning": score < 0.25   # less than 25% keyword overlap = suspicious
    }


# ──────────────────────────────────────────
# JSON EXTRACTOR — handles all edge cases
# ──────────────────────────────────────────

def extract_json(raw: str) -> dict | None:
    """
    Extracts first valid JSON object from raw string.
    Handles markdown fences, extra text, multiple JSONs.
    """
    if not raw or not raw.strip():
        return None

    # Remove markdown code fences
    clean = raw
    for fence in ["```json", "```python", "```"]:
        clean = clean.replace(fence, " ")

    # Find first complete JSON object
    depth  = 0
    start  = -1
    result = ""

    for i, char in enumerate(clean):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                result = clean[start:i+1]
                break

    if not result:
        return None

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ──────────────────────────────────────────

def evaluate_response(query: str, response: str,
                      context: str = None) -> dict:
    """
    Uses Llama 70B as judge to evaluate response quality.
    Runs asynchronously after response is sent to user.

    Improvements over previous version:
    - Bug fix: uses raw_output["text"] not raw_output
    - Hallucination detection via judge prompt
    - Retrieval quality pre-check
    - Hallucination penalty in quality score

    Args:
        query:    original user question
        response: model answer
        context:  RAG chunks if available

    Returns:
        dict with scores, quality_score, hallucinations, retrieval_warning
    """

    # ── Pre-check: retrieval quality ──
    retrieval_check = check_retrieval_quality(query, context or "")

    if retrieval_check["warning"]:
        print(f"⚠️  Retrieval quality warning — "
              f"low keyword overlap ({retrieval_check['score']}) "
              f"between query and context")

    # ── Build prompt ──
    context_section = CONTEXT_SECTION.format(context=context) \
                      if context else NO_CONTEXT_NOTE

    prompt = JUDGE_PROMPT.format(
        query           = query,
        response        = response,
        context_section = context_section
    )

    try:
        # ── BUG FIX: invoke_model returns dict, extract .text ──
        raw_output = invoke_model("haiku", prompt)
        raw_text   = raw_output["text"] if isinstance(raw_output, dict) \
                     else str(raw_output)

        scores = extract_json(raw_text)

        if scores is None:
            return _fallback(f"No JSON found in output: {raw_text[:150]}")

        relevance    = float(scores.get("relevance",    0.0))
        correctness  = float(scores.get("correctness",  0.0))
        completeness = float(scores.get("completeness", 0.0))
        reasoning    = scores.get("reasoning", "")

        # Hallucination flags — new field
        hallucination_flags = scores.get("hallucination_flags", [])
        if not isinstance(hallucination_flags, list):
            hallucination_flags = []

        hallucination_count = len(hallucination_flags)

        if hallucination_count > 0:
            print(f"⚠️  Hallucination flags detected ({hallucination_count}): "
                  f"{hallucination_flags}")

        quality_score = calculate_quality_score(
            relevance, correctness, completeness, hallucination_count
        )

        return {
            "relevance":           relevance,
            "correctness":         correctness,
            "completeness":        completeness,
            "quality_score":       quality_score,
            "reasoning":           reasoning,
            "hallucination_flags": hallucination_flags,
            "retrieval_score":     retrieval_check["score"],
            "retrieval_warning":   retrieval_check["warning"],
            "success":             True
        }

    except Exception as e:
        return _fallback(str(e))


def _fallback(reason: str) -> dict:
    """Returns neutral fallback scores when evaluation fails."""
    print(f"⚠️  Evaluation fallback: {reason}")
    return {
        "relevance":           0.5,
        "correctness":         0.5,
        "completeness":        0.5,
        "quality_score":       0.5,
        "reasoning":           reason,
        "hallucination_flags": [],
        "retrieval_score":     1.0,
        "retrieval_warning":   False,
        "success":             False
    }


# ──────────────────────────────────────────
# GOLDEN TEST DATASET
# Known correct Q&A pairs from your documents.
# Run this to validate pipeline after code changes.
# ──────────────────────────────────────────

GOLDEN_TESTS = [
    # Add your real document Q&A pairs here
    # Format: (query, correct_answer_keywords, context_snippet)
    {
        "query":    "What is Python?",
        "response": "Python is a high-level interpreted programming language "
                    "known for readability and versatility.",
        "context":  None,
        "expect_quality_above": 0.75
    },
    {
        "query":    "What is Python?",
        "response": "Python is a type of snake found in tropical regions.",
        "context":  None,
        "expect_quality_above": None,   # expect LOW score
        "expect_quality_below": 0.50
    },
]


# ──────────────────────────────────────────
# TEST RUNNER
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Evaluation Engine Test ──\n")

    # Test 1 — Good response (no context)
    result1 = evaluate_response(
        query    = "What is Python?",
        response = "Python is a high-level interpreted programming language "
                   "known for readability and versatility. It supports multiple "
                   "paradigms including OOP and functional programming.",
        context  = None
    )
    print("Test 1 — Good response:")
    print(f"   Relevance:    {result1['relevance']}")
    print(f"   Correctness:  {result1['correctness']}")
    print(f"   Completeness: {result1['completeness']}")
    print(f"   Quality:      {result1['quality_score']}")
    print(f"   Hallucinations: {result1['hallucination_flags']}")
    print(f"   Reasoning:    {result1['reasoning']}")
    print()

    # Test 2 — Bad response (no context)
    result2 = evaluate_response(
        query    = "What is Python?",
        response = "Python is a type of snake found in tropical regions.",
        context  = None
    )
    print("Test 2 — Bad response:")
    print(f"   Relevance:    {result2['relevance']}")
    print(f"   Correctness:  {result2['correctness']}")
    print(f"   Completeness: {result2['completeness']}")
    print(f"   Quality:      {result2['quality_score']}")
    print(f"   Hallucinations: {result2['hallucination_flags']}")
    print(f"   Reasoning:    {result2['reasoning']}")
    print()

    # Test 3 — RAG response faithful to context
    result3 = evaluate_response(
        query    = "What is the leave policy?",
        response = "Employees are entitled to 20 days of paid leave per year "
                   "with rollover up to 30 days.",
        context  = "Company policy states employees receive 20 days annual "
                   "leave with rollover up to 30 days."
    )
    print("Test 3 — RAG faithful response:")
    print(f"   Quality:        {result3['quality_score']}")
    print(f"   Hallucinations: {result3['hallucination_flags']}")
    print(f"   Retrieval score:{result3['retrieval_score']}")
    print()

    # Test 4 — RAG response with hallucination
    result4 = evaluate_response(
        query    = "What is the leave policy?",
        response = "Employees get 30 days leave, free health insurance, "
                   "and a company car.",
        context  = "Company policy states employees receive 20 days annual leave."
    )
    print("Test 4 — RAG hallucination response:")
    print(f"   Quality:        {result4['quality_score']}")
    print(f"   Hallucinations: {result4['hallucination_flags']}")
    print()

    # Sanity check
    print("── Sanity Check ──")
    good = result1["quality_score"]
    bad  = result2["quality_score"]
    print(f"Good response score: {good}")
    print(f"Bad response score:  {bad}")
    if good > bad:
        print("✅ Evaluator correctly scores good > bad")
    else:
        print("❌ Scoring logic needs review")

    print("\n✅ Evaluation engine test complete!")
