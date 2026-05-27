# core/intent_detector.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ──────────────────────────────────────────
# ORG MARKERS
# Words that signal the query is about THIS company.
# At least one must be present for HR topics to
# trigger "specific" intent.
# ──────────────────────────────────────────

ORG_MARKERS = [
    "our",
    "my",
    "we",
    "us",
    "here",
    "this company",
    "the company",
    "our company",
    "organization",
    "firm",
    "info services",
    "infoservices",
    "as per",
    "according to",
    "what does it say",
    "what does the document",
    "what does our",
]

# ──────────────────────────────────────────
# HR TOPICS
# These alone are NOT enough to trigger specific.
# Must appear alongside an ORG_MARKER.
# ──────────────────────────────────────────

HR_TOPICS = [
    "leave",
    "salary",
    "benefits",
    "allowance",
    "reimbursement",
    "resignation",
    "notice period",
    "policy",
    "policies",
    "clause",
    "clauses",
    "section",
    "guideline",
    "handbook",
    "contract",
    "agreement",
    "appraisal",
    "probation",
    "onboarding",
    "offboarding",
]

# ──────────────────────────────────────────
# ALWAYS SPECIFIC — no org marker needed
# These phrases are inherently company-specific
# ──────────────────────────────────────────

ALWAYS_SPECIFIC = [
    "our leave policy",
    "our salary",
    "our benefits",
    "our notice period",
    "our resignation",
    "our handbook",
    "our policy",
    "our contract",
    "our guidelines",
    "company policy",
    "company handbook",
    "hr policy",
    "employee handbook",
    "as per the policy",
    "as per our",
    "according to the policy",
    "according to our",
    "what does the handbook",
    "what does our policy",
]

# ──────────────────────────────────────────
# ALWAYS GENERAL — even if HR words present
# ──────────────────────────────────────────

ALWAYS_GENERAL = [
    "in general",
    "generally speaking",
    "typically",
    "usually",
    "across the industry",
    "in most companies",
    "what is meant by",
    "define ",
    "definition of",
]

# ──────────────────────────────────────────
# SEMANTIC ANCHORS
# ──────────────────────────────────────────

GENERAL_ANCHOR  = "What does this term mean in general?"
SPECIFIC_ANCHOR = "What does our company policy say about this?"

# ──────────────────────────────────────────
# MODEL LOAD
# ──────────────────────────────────────────

print("Loading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

GENERAL_EMBEDDING  = embedder.encode([GENERAL_ANCHOR])
SPECIFIC_EMBEDDING = embedder.encode([SPECIFIC_ANCHOR])
print("Intent detector ready.")


# ──────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────

def detect_intent(query: str) -> dict:
    """
    Detects whether query is general knowledge or company-specific.

    Logic:
    1. ALWAYS_GENERAL phrases → immediately general
    2. ALWAYS_SPECIFIC phrases → immediately specific
    3. ORG_MARKER + HR_TOPIC both present → specific
    4. ORG_MARKER alone → specific (asking about company context)
    5. HR_TOPIC alone (no org marker) → general
       e.g. "benefits of freelancing" stays general
    6. Semantic similarity as tiebreaker

    Returns:
        dict with intent (general/specific/unknown) and confidence
    """
    query_lower = query.lower()

    # ── Rule 1: Always general overrides ──
    for phrase in ALWAYS_GENERAL:
        if phrase in query_lower:
            return {
                "intent":     "general",
                "confidence": 0.95
            }

    # ── Rule 2: Always specific phrases ──
    for phrase in ALWAYS_SPECIFIC:
        if phrase in query_lower:
            return {
                "intent":     "specific",
                "confidence": 0.95
            }

    # ── Rule 3: Check org marker presence ──
    has_org_marker = any(marker in query_lower for marker in ORG_MARKERS)
    has_hr_topic   = any(topic  in query_lower for topic  in HR_TOPICS)

    # Org marker + HR topic → specific
    if has_org_marker and has_hr_topic:
        return {
            "intent":     "specific",
            "confidence": 0.90
        }

    # Org marker alone (e.g. "our process", "our system") → specific
    if has_org_marker:
        return {
            "intent":     "specific",
            "confidence": 0.75
        }

    # HR topic without org marker → do NOT classify as specific
    # e.g. "benefits of freelancing", "leave it as is", "salary negotiation tips"
    # Fall through to semantic check

    # ── Rule 4: Semantic similarity as tiebreaker ──
    query_embedding = embedder.encode([query])
    general_sim     = float(cosine_similarity(query_embedding, GENERAL_EMBEDDING)[0][0])
    specific_sim    = float(cosine_similarity(query_embedding, SPECIFIC_EMBEDDING)[0][0])

    # If HR topic present but no org marker, bias toward general
    if has_hr_topic and not has_org_marker:
        general_sim += 0.15  # nudge toward general

    if specific_sim > general_sim and specific_sim > 0.35:
        return {
            "intent":     "specific",
            "confidence": round(specific_sim, 2)
        }
    elif general_sim > specific_sim and general_sim > 0.25:
        return {
            "intent":     "general",
            "confidence": round(general_sim, 2)
        }
    else:
        return {
            "intent":     "unknown",
            "confidence": 0.0
        }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # query                                              expected
        ("What is Python?",                                  "general"),
        ("What are the benefits of being a freelancer?",     "general"),
        ("What are some salary negotiation tips?",           "general"),
        ("How does leave management work in HR?",            "general"),
        ("Explain machine learning generally",               "general"),
        ("Define artificial intelligence",                   "general"),
        ("What is our leave policy?",                        "specific"),
        ("What are our salary benefits?",                    "specific"),
        ("What does our contract say about resignation?",    "specific"),
        ("As per the policy what is the notice period?",     "specific"),
        ("According to our handbook what are the rules?",    "specific"),
        ("Is this related to info services documents?",      "specific"),
        ("What does the company say about probation?",       "specific"),
    ]

    passed = 0
    failed = 0

    print("\n── Intent Detection Results ──\n")

    for query, expected in test_cases:
        result  = detect_intent(query)
        ok      = result["intent"] == expected
        icon    = "✅" if ok else "❌"

        if ok:
            passed += 1
        else:
            failed += 1

        print(f"{icon} {query}")
        print(f"   Expected : {expected}")
        print(f"   Got      : {result['intent']} (confidence={result['confidence']})")
        print()

    print(f"── Results: {passed} passed / {failed} failed ──")
