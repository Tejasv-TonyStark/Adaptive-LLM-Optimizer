"""Validated runtime settings."""
import os
from dotenv import load_dotenv
load_dotenv()
def fraction(name, default):
    value = float(os.getenv(name, str(default)))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value
EVALUATION_SAMPLE_RATE = fraction("EVALUATION_SAMPLE_RATE", 0.2)
EXPLORATION_RATE = fraction("EXPLORATION_RATE", 0.05)
MAX_EXPLORATION_COST_USD = float(os.getenv("MAX_EXPLORATION_COST_USD", "0.002"))
if not 0 < MAX_EXPLORATION_COST_USD < float("inf"):
    raise ValueError("MAX_EXPLORATION_COST_USD must be positive and finite")
ROUTING_POLICY = os.getenv("ROUTING_POLICY", "static")
if ROUTING_POLICY not in {"static", "adaptive"}:
    raise ValueError("ROUTING_POLICY must be static or adaptive")
