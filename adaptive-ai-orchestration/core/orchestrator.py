from core.intent_detector import detect_intent
from core.complexity_analyzer import analyze_complexity
def select_strategy(complexity):
    return "fast" if complexity == "low" else "reasoning"
def orchestrate(query):
    intent = detect_intent(query)
    analysis = analyze_complexity(query, intent["intent"])
    strategy = select_strategy(analysis["complexity"])
    return {**intent, **analysis, "strategy": strategy,
            "execution_strategy": "rag" if analysis["retrieval_needed"] else strategy,
            "routing_reasons": [intent["reason"], *analysis["reasons"]]}
