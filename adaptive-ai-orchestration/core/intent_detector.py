# core/intent_detector.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ──────────────────────────────────────────
# KEYWORDS
# ──────────────────────────────────────────

SPECIFIC_KEYWORDS = [
    "our", "my", "this company", "here",
    "as per", "according to", "what does it say",
    "organization", "firm", "we", "us",
    "policy", "policies", "clause", "section",
    "guideline", "reimbursement", "leave", "salary",
    "benefits", "allowance", "resignation", "notice period"
]

GENERAL_KEYWORDS = [
    "generally", "typically", "usually",
    "in general", "define", "what is meant by",
    "explain what", "tell me about", "what is",
    "how does", "what are"
]

# ──────────────────────────────────────────
# ANCHOR SENTENCES FOR SEMANTIC SIMILARITY
# ──────────────────────────────────────────

GENERAL_ANCHOR = "What does this term mean in general?"
SPECIFIC_ANCHOR = "What does our company policy say about this?"

# ──────────────────────────────────────────
# MODEL LOAD
# ──────────────────────────────────────────

print("Loading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Pre compute anchor embeddings once at startup
GENERAL_EMBEDDING  = embedder.encode([GENERAL_ANCHOR])
SPECIFIC_EMBEDDING = embedder.encode([SPECIFIC_ANCHOR])
print("Intent detector ready.")


# ──────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────

def detect_intent(query: str) -> dict:
    """
    Detects whether the query is general knowledge
    or company specific information.

    Returns:
        dict with intent (general/specific/unknown) and confidence
    """
    query_lower = query.lower()

    # ── Step 1: Keyword matching ──
    specific_hits = sum(1 for kw in SPECIFIC_KEYWORDS if kw in query_lower)
    general_hits  = sum(1 for kw in GENERAL_KEYWORDS  if kw in query_lower)

    # ── Step 2: Semantic similarity ──
    query_embedding = embedder.encode([query])

    general_sim  = cosine_similarity(query_embedding, GENERAL_EMBEDDING)[0][0]
    specific_sim = cosine_similarity(query_embedding, SPECIFIC_EMBEDDING)[0][0]

    # ── Step 3: Combined scoring ──
    # Keywords weighted 60%, semantic similarity 40%
    general_score  = (general_hits  * 0.6) + (float(general_sim)  * 0.4)
    specific_score = (specific_hits * 0.6) + (float(specific_sim) * 0.4)

    # ── Step 4: Decision ──
    if specific_score > general_score and specific_score > 0.3:
        intent     = "specific"
        confidence = round(float(specific_score), 2)
    elif general_score > specific_score and general_score > 0.3:
        intent     = "general"
        confidence = round(float(general_score), 2)
    else:
        intent     = "unknown"
        confidence = 0.0

    return {
        "intent":     intent,
        "confidence": confidence
    }


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is Python?",
        "What is our leave policy?",
        "Explain machine learning generally",
        "What does our contract say about resignation?",
        "Define artificial intelligence",
        "What are our salary benefits?"
    ]

    print("\n── Intent Detection Results ──\n")
    for query in test_queries:
        result = detect_intent(query)
        print(f"Query:      {query}")
        print(f"Intent:     {result['intent']}")
        print(f"Confidence: {result['confidence']}")
        print()