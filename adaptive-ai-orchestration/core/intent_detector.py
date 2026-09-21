"""Explainable single-turn rules; confidence is heuristic, not calibrated."""
from core.text_rules import contains
GENERAL = ("in general", "generally speaking", "typically", "usually",
           "in most companies", "across the industry", "define", "definition of")
DOCUMENT = ("our handbook", "employee handbook", "company handbook", "our policy",
            "our contract", "our guidelines", "company policy", "hr policy",
            "according to our", "as per our", "as per the policy",
            "according to the policy", "uploaded document", "uploaded documents")
ORG = ("our", "my", "this company", "the company", "here", "infoservices", "info services")
HR = ("leave", "salary", "benefits", "allowance", "reimbursement", "resignation",
      "notice period", "policy", "policies", "contract", "handbook", "probation",
      "onboarding", "offboarding", "appraisal", "working hours", "work from home",
      "provident fund", "shift timings", "vacation", "sick", "earned leave")
DIRECT_HR = ("notice period", "sick leave", "sick leaves", "vacation leave",
             "vacation leaves", "earned leave", "earned leaves", "probation period",
             "monthly salary", "working hours", "work from home", "provident fund",
             "shift timings", "daily allowance", "performance appraisal")
def detect_intent(query: str) -> dict:
    text = query.casefold()
    explicit = next((p for p in DOCUMENT if contains(text, p)), None)
    if explicit:
        return dict(intent="specific", confidence=0.95, reason=f"document reference: {explicit}")
    general = next((p for p in GENERAL if contains(text, p)), None)
    if general:
        return dict(intent="general", confidence=0.9, reason=f"general qualifier: {general}")
    if any(contains(text, p) for p in ORG) and any(contains(text, p) for p in HR):
        return dict(intent="specific", confidence=0.85, reason="organization marker and policy topic")
    if any(contains(text, p) for p in DIRECT_HR):
        return dict(intent="specific", confidence=0.65, reason="policy topic in company Q&A domain")
    return dict(intent="general", confidence=0.5, reason="no document reference; single-turn default")
