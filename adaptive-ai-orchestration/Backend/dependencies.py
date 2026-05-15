# backend/dependencies.py

from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from dotenv import load_dotenv
import os

load_dotenv()

# ──────────────────────────────────────────
# API KEY SETUP
# ──────────────────────────────────────────

# The header name where API key must be sent
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load valid API keys from .env
# Multiple keys supported — comma separated
VALID_API_KEYS = os.getenv("API_KEYS", "").split(",")


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Checks if the request has a valid API key in the header.
    Called automatically by FastAPI on every protected endpoint.
    Raises 403 if key is missing or invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="No API key provided. Include X-API-Key in your request header."
        )
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )
    return api_key


# ──────────────────────────────────────────
# RATE LIMITER SETUP
# ──────────────────────────────────────────

# Limits requests by IP address
# 10 requests per minute per IP
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    # Test: check API keys loaded correctly
    if VALID_API_KEYS and VALID_API_KEYS[0]:
        print(f"✅ API keys loaded: {len(VALID_API_KEYS)} key(s) found")
    else:
        print("⚠️  No API keys found in .env — add API_KEYS to your .env file")

    print("✅ dependencies.py working correctly!")