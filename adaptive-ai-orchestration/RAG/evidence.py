"""Fail-closed extractive answers: only exact, cited passages reach the user.

This validates source membership, not semantic relevance or completeness.
"""
import json
import re

ABSTENTION = "Not found in the available documents."
MAX_EVIDENCE_ITEMS = 2
MAX_EVIDENCE_CHARS = 600
MAX_QUOTE_CHARS = 350
QUERY_STOPWORDS = {"a", "an", "and", "are", "about", "for", "how", "is", "of", "our",
                   "policy", "the", "to", "what", "when", "where", "which", "who", "why"}

def _query_terms(text):
    return {word for word in re.findall(r"[a-z0-9]{3,}", text.casefold())
            if word not in QUERY_STOPWORDS}

def extractive_fallback(query, source_chunks):
    """Return one relevant, exact declarative sentence from retrieved text.

    This is used only after provider evidence fails validation. It cannot invent
    wording: the returned quote is a substring of one retrieved source chunk.
    """
    terms = _query_terms(query)
    candidates = []
    for identifier, chunk in enumerate(source_chunks or [], 1):
        for fragment in re.split(r"(?<=[.!?])\s+", chunk.get("text", "")):
            quote = fragment.rsplit("\n", 1)[-1].strip()
            if (not quote or quote.endswith("?") or len(quote) > MAX_QUOTE_CHARS
                    or not quote.endswith((".", "!"))):
                continue
            score = len(terms & set(re.findall(r"[a-z0-9]{3,}", fragment.casefold())))
            if score:
                candidates.append((score, -len(quote), identifier, quote))
    if not candidates:
        return None
    _, _, identifier, quote = max(candidates)
    return f"{quote} [{identifier}]"

def render_evidence(raw, context, source_chunks=None, query=None):
    text = raw.strip()
    if text in {ABSTENTION, "Not found in the provided documents."}:
        return ABSTENTION, True
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"answerable", "evidence"}:
        raise ValueError("Expected answerable and evidence")
    if type(data["answerable"]) is not bool or not isinstance(data["evidence"], list):
        raise ValueError("Invalid evidence types")
    if data["answerable"] is False:
        if data["evidence"]:
            raise ValueError("Abstention cannot contain evidence")
        return ABSTENTION, True
    if not 1 <= len(data["evidence"]) <= MAX_EVIDENCE_ITEMS:
        raise ValueError("Expected one or two supporting passages")
    passages = {}
    if source_chunks is not None:
        # Authoritative metadata avoids treating fake source markers in PDF text as citations.
        passages = {n: chunk["text"] for n, chunk in enumerate(source_chunks, 1)}
    else:
        for part in context.split("\n\n---\n\n"):
            match = re.match(r"\[(\d+)\] [^\n]+\n([\s\S]+)", part)
            if match:
                passages[int(match[1])] = match[2]
    lines, quotes = [], set()
    total_chars = 0
    for item in data["evidence"]:
        if not isinstance(item, dict) or set(item) != {"source_id", "quote"}:
            raise ValueError("Invalid citation fields")
        identifier, quote = item["source_id"], item["quote"]
        if type(identifier) is not int or identifier not in passages:
            raise ValueError("Citation outside supplied context")
        if (not isinstance(quote, str) or not quote.strip() or quote != quote.strip() or quote.endswith("?")
                or len(quote) > MAX_QUOTE_CHARS or quote not in passages[identifier]):
            raise ValueError("Quote is not verbatim source evidence")
        if query and _query_terms(query) and not (_query_terms(query) & _query_terms(quote)):
            raise ValueError("Quote is not relevant to the question")
        line = f"{quote} [{identifier}]"
        # The bundled handbooks share much of their text. Display the evidence
        # once rather than returning the same answer for each matching source.
        if quote not in quotes:
            total_chars += len(line)
            if total_chars > MAX_EVIDENCE_CHARS:
                raise ValueError("Evidence exceeds response length limit")
            lines.append(line)
            quotes.add(quote)
    return "\n\n".join(lines), False
