import re
def contains(text: str, phrase: str) -> bool:
    """Whole phrases, whitespace tolerant, never word fragments."""
    pattern = r"\s+".join(re.escape(word) for word in phrase.strip().split())
    return bool(re.search(r"(?<!\w)" + pattern + r"(?!\w)", text.casefold()))
