"""Fail-closed extractive answers: only exact, cited passages reach the user.

This validates source membership, not semantic relevance or completeness.
"""
import json
import re

ABSTENTION = "Not found in the available documents."

def render_evidence(raw, context):
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
    if not 1 <= len(data["evidence"]) <= 5:
        raise ValueError("Expected one to five supporting passages")
    passages = {}
    for part in context.split("\n\n---\n\n"):
        match = re.match(r"\[(\d+)\] [^\n]+\n([\s\S]+)", part)
        if match:
            passages[int(match[1])] = match[2]
    lines = []
    for item in data["evidence"]:
        if not isinstance(item, dict) or set(item) != {"source_id", "quote"}:
            raise ValueError("Invalid citation fields")
        identifier, quote = item["source_id"], item["quote"]
        if type(identifier) is not int or identifier not in passages:
            raise ValueError("Citation outside supplied context")
        if not isinstance(quote, str) or not quote.strip() or quote != quote.strip() or quote not in passages[identifier]:
            raise ValueError("Quote is not verbatim source evidence")
        line = f"{quote} [{identifier}]"
        if line not in lines:
            lines.append(line)
    return "\n\n".join(lines), False
