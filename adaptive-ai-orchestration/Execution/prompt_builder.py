# execution/prompt_builder.py

# ──────────────────────────────────────────
# SYSTEM PROMPTS PER STRATEGY
# ──────────────────────────────────────────

SYSTEM_PROMPTS = {
    "fast": """You are a helpful assistant.
Answer the question directly and concisely.
Keep your response under 3 sentences.
Do not over-explain.""",

    "reasoning": """You are an expert analytical assistant.
Think step by step before answering.
Provide a well-structured, detailed response.
Use clear reasoning and explain your logic.""",

    "rag": """You are a helpful assistant with access to company documents.
Answer ONLY based on the provided document context.
If the answer is not in the context, say: 'I could not find this in the provided documents.'
Do not use outside knowledge for company-specific questions.
Be precise and cite relevant sections when possible."""
}


# ──────────────────────────────────────────
# PROMPT BUILDERS
# ──────────────────────────────────────────

def build_prompt(query: str, strategy: str,
                 context: str = None) -> str:
    """
    Builds the full prompt for a given query and strategy.

    For RAG strategy: injects retrieved document chunks as context.
    For fast/reasoning: direct question with system instructions.

    Args:
        query:    user question
        strategy: fast / reasoning / rag
        context:  retrieved document chunks (RAG only)

    Returns:
        str — complete prompt ready to send to model
    """
    system = SYSTEM_PROMPTS.get(strategy, SYSTEM_PROMPTS["fast"])

    # RAG prompt — includes document context
    if strategy == "rag" and context:
        prompt = f"""{system}

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---

User Question: {query}

Answer:"""

    # RAG requested but no context found
    elif strategy == "rag" and not context:
        prompt = f"""{system}

Note: No relevant documents were found for this query.
Answer based on general knowledge if appropriate.

User Question: {query}

Answer:"""

    # Fast or reasoning — direct prompt
    else:
        prompt = f"""{system}

User Question: {query}

Answer:"""

    return prompt


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Prompt Builder Test ──\n")

    # Test 1 — Fast prompt
    p1 = build_prompt("What is Python?", "fast")
    print("FAST PROMPT:")
    print(p1)
    print()

    # Test 2 — Reasoning prompt
    p2 = build_prompt(
        "Compare transformers vs RNN architectures",
        "reasoning"
    )
    print("REASONING PROMPT:")
    print(p2)
    print()

    # Test 3 — RAG prompt with context
    p3 = build_prompt(
        "What is our leave policy?",
        "rag",
        context="Employees are entitled to 18 days of annual leave per year."
    )
    print("RAG PROMPT WITH CONTEXT:")
    print(p3)
    print()

    # Test 4 — RAG prompt without context
    p4 = build_prompt(
        "What is our leave policy?",
        "rag",
        context=None
    )
    print("RAG PROMPT WITHOUT CONTEXT:")
    print(p4)
    print()

    print("✅ Prompt builder working correctly!")