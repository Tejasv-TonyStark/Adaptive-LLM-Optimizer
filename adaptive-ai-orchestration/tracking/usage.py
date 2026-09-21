"""Estimated USD accounting. Configure rates for your region/provider contract.
Defaults are illustrative, not a current-price guarantee. Failed calls without
provider usage have unknown cost, never a claimed zero bill.
"""
import json
import os
from dotenv import load_dotenv
load_dotenv()
RATES = json.loads(os.getenv("MODEL_PRICING_JSON", "null")) or {
    "nova-micro": {"input": 0.000035, "output": 0.000140},
    "llama3-8b": {"input": 0.000220, "output": 0.000220},
    "llama3-70b": {"input": 0.000720, "output": 0.000720},
    "titan-embed-v2": {"input": 0.000020, "output": 0.0},
}
def estimate_tokens(text):
    return max(1, (len(text) + 3) // 4)
def calculate_cost(model, input_tokens, output_tokens):
    rate = RATES[model]
    return round((input_tokens * rate["input"] + output_tokens * rate["output"]) / 1000, 10)
def usage_record(model, stage, prompt, result, latency_ms, error=None):
    if error is not None:
        return dict(model=model, stage=stage, status="failed", input_tokens=None,
                    output_tokens=None, estimated_cost=None, estimated=True,
                    latency_ms=latency_ms, error=type(error).__name__)
    inp, out = result.get("input_tokens"), result.get("output_tokens")
    estimated = inp is None or out is None
    inp = estimate_tokens(prompt) if inp is None else inp
    out = estimate_tokens(result.get("text", "")) if out is None else out
    return dict(model=model, stage=stage, status="completed", input_tokens=inp,
                output_tokens=out, estimated_cost=calculate_cost(model, inp, out),
                estimated=estimated, latency_ms=latency_ms, error=None)
