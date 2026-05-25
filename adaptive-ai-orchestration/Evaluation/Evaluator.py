# evaluation/evaluator.py

import json
from Execution.Bedrock_client import invoke_model

# ──────────────────────────────────────────
# JUDGE PROMPT — stricter instructions
# ──────────────────────────────────────────

JUDGE_PROMPT = """You are an AI response evaluator.

Evaluate this AI response and return scores.

Question: {query}

AI Response: {response}

{context_section}

INSTRUCTIONS:
- Score each criterion from 0.0 to 1.0
- relevance: does the response address the question?
- correctness: is the information accurate?
- completeness: is the answer complete?
- Return ONLY the JSON below, nothing else, no extra text

{{"relevance": 0.0, "correctness": 0.0, "completeness": 0.0, "reasoning": "brief reason"}}"""

CONTEXT_SECTION = """Document Context (ground truth for correctness):
{context}"""


# ──────────────────────────────────────────
# SCORING FORMULA
# ──────────────────────────────────────────

def calculate_quality_score(relevance: float,
                             correctness: float,
                             completeness: float) -> float:
    """
    quality = 0.40*correctness + 0.35*relevance + 0.25*completeness
    """
    return round(
        (0.40 * correctness) +
        (0.35 * relevance)   +
        (0.25 * completeness),
        4
    )


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

    # Remove all markdown code fences first
    clean = raw
    for fence in ["```json", "```python", "```"]:
        clean = clean.replace(fence, " ")

    # Find ALL JSON objects in the text
    # Take the FIRST one only
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

    Args:
        query:    original user question
        response: model answer
        context:  RAG chunks if available

    Returns:
        dict with scores and quality_score
    """
    context_section = CONTEXT_SECTION.format(context=context) \
                      if context else ""

    prompt = JUDGE_PROMPT.format(
        query           = query,
        response        = response,
        context_section = context_section
    )

    try:
        raw_output = invoke_model("haiku", prompt)
        scores     = extract_json(raw_output)

        if scores is None:
            return _fallback(f"No JSON found in output: {raw_output[:100]}")

        relevance    = float(scores.get("relevance",    0.0))
        correctness  = float(scores.get("correctness",  0.0))
        completeness = float(scores.get("completeness", 0.0))
        reasoning    = scores.get("reasoning", "")

        quality_score = calculate_quality_score(
            relevance, correctness, completeness
        )

        return {
            "relevance":     relevance,
            "correctness":   correctness,
            "completeness":  completeness,
            "quality_score": quality_score,
            "reasoning":     reasoning,
            "success":       True
        }

    except Exception as e:
        return _fallback(str(e))


def _fallback(reason: str) -> dict:
    """Returns neutral fallback scores when evaluation fails."""
    print(f"⚠️  Evaluation fallback: {reason}")
    return {
        "relevance":     0.5,
        "correctness":   0.5,
        "completeness":  0.5,
        "quality_score": 0.5,
        "reasoning":     reason,
        "success":       False
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Evaluation Engine Test ──\n")

    # Test 1 — Good response
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
    print(f"   Reasoning:    {result1['reasoning']}")
    print()

    # Test 2 — Bad response
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
    print(f"   Reasoning:    {result2['reasoning']}")
    print()

    # Test 3 — RAG response with context
    result3 = evaluate_response(
        query    = "What is the leave policy?",
        response = "Employees are entitled to 20 days of paid leave per year.",
        context  = "Company policy states employees receive 20 days annual "
                   "leave with rollover up to 30 days."
    )
    print("Test 3 — RAG response:")
    print(f"   Relevance:    {result3['relevance']}")
    print(f"   Correctness:  {result3['correctness']}")
    print(f"   Completeness: {result3['completeness']}")
    print(f"   Quality:      {result3['quality_score']}")
    print(f"   Reasoning:    {result3['reasoning']}")
    print()

    # Verify scores make sense
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