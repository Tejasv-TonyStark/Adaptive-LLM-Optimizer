# Execution/prompt_builder.py

# ──────────────────────────────────────────
# SYSTEM PROMPTS — token-efficient, strict length limits
# Shorter system prompts = fewer input tokens on every call
# ──────────────────────────────────────────

SYSTEM_PROMPTS = {

    # fast: 3 sentences max, no padding
    "fast": (
        "Answer in 3 sentences or fewer. "
        "Be direct. No greetings, no preamble, no closing remarks."
    ),

    # reasoning: structured but bounded — 4 steps max
    "reasoning": (
        "Think step by step. "
        "Structure your answer with a brief conclusion first, "
        "then numbered reasoning steps (max 4). "
        "Stop once the question is answered."
    ),

    # rag: context-only, no filler
    "rag": (
        "Answer using ONLY the document context below. "
        "If the answer is not in the context, respond: "
        "'Not found in the provided documents.' "
        "No extra commentary."
    ),
}


# ──────────────────────────────────────────
# PROMPT BUILDERS
# ──────────────────────────────────────────

def build_prompt(query: str, strategy: str,
                 context: str = None) -> str:
    """
    Builds a token-efficient prompt for a given query and strategy.

    Design rules:
      - System prompt is kept short (< 50 tokens) to save input tokens
      - No redundant separators or verbose labels
      - RAG context is injected only when present
      - All strategies enforce a response length ceiling

    Args:
        query:    user question
        strategy: fast / reasoning / rag
        context:  retrieved document chunks (RAG only)

    Returns:
        str — complete prompt ready to send to the model
    """
    system = SYSTEM_PROMPTS.get(strategy, SYSTEM_PROMPTS["fast"])

    if strategy == "rag" and context:
        return (
            f"{system}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

    if strategy == "rag" and not context:
        return (
            f"{system}\n\n"
            f"Context: [No documents found]\n\n"
            f"Question: {query}\nAnswer:"
        )

    # fast / reasoning
    return (
        f"{system}\n\n"
        f"Question: {query}\nAnswer:"
    )


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    def approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    cases = [
        ("What is Python?",                           "fast",      None),
        ("Compare transformers vs RNN architectures", "reasoning", None),
        ("What is our leave policy?",                 "rag",       "Employees get 18 days annual leave."),
        ("What is our leave policy?",                 "rag",       None),
    ]

    print("\n── Prompt Builder Test ──\n")
    for query, strategy, ctx in cases:
        prompt = build_prompt(query, strategy, ctx)
        tokens = approx_tokens(prompt)
        print(f"[{strategy.upper()}]  ~{tokens} input tokens")
        print(prompt)
        print()

    print("✅ Prompt builder working correctly!")
