"""Transparent baseline, not a trained difficulty classifier.
Task requirements take priority over topic names; scores are not probabilities.
"""
import re
from core.text_rules import contains
HIGH_TASKS = ("prove", "derive", "critically analyze", "critically evaluate",
              "find contradictions", "reconcile", "root cause analysis",
              "mathematical foundations", "failure modes")
MEDIUM_TASKS = ("explain", "how does", "how do", "compare", "contrast", "summarize",
                "summarise", "summary", "overview", "analyze", "analyse", "evaluate",
                "implement", "write", "design", "build", "debug")
CONSTRAINTS = ("distributed", "concurrent", "deadlock", "retries", "fault tolerant",
               "thread safe", "across contracts", "across multiple", "in detail",
               "precedence constraints", "optimal schedule", "race conditions")
def _check_rag(query_lower, intent):
    return intent == "specific"
def analyze_complexity(query: str, intent: str) -> dict:
    text = query.casefold().strip()
    reasons = [p for p in HIGH_TASKS if contains(text, p)]
    tasks = [p for p in MEDIUM_TASKS if contains(text, p)]
    constraints = [p for p in CONSTRAINTS if contains(text, p)]
    uncertain = False
    # A narrow, verifiable elementary task; do not downgrade arbitrary proofs.
    elementary = re.fullmatch(r"(?:prove|show|demonstrate) that \d{1,6} is (?:an? )?(?:even|odd)(?: number)?[.!?]?", text)
    if elementary:
        complexity, score, reasons = "low", 0, ["elementary integer parity"]
    elif reasons or len(constraints) >= 2 or (
        contains(text, "compare") and (contains(text, "architectures") or contains(text, "in detail"))
    ):
        complexity, score = "high", 5
        reasons += tasks + constraints
    elif re.match(r"^(define\b|what is meant by\b|what is the definition of\b)", text):
        complexity, score, reasons = "low", 0, ["definition request"]
    elif intent == "specific" and tasks and all(t in {"how do", "how does"} for t in tasks) and not constraints:
        complexity, score, reasons = "low", 0, ["document procedure lookup"]
    elif tasks:
        complexity, score, reasons = "medium", 2, tasks + constraints
    elif re.match(r"^(what (?:is|are|does)\b|who\b|when\b|where\b|how (?:many|much)\b)", text) or (
        intent == "specific" and not tasks and not constraints
    ):
        complexity, score, reasons = "low", 0, ["factual/default request"]
    else:
        complexity, score, reasons = "medium", 2, ["unrecognized task; conservative route"]
        uncertain = True
    return dict(complexity=complexity, score=score, retrieval_needed=_check_rag(text, intent),
                trigger="; ".join(reasons), reasons=reasons, uncertain=uncertain)
