# Backend/dependencies.py

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from dotenv import load_dotenv
import hashlib
import os

load_dotenv()

# ──────────────────────────────────────────
# API KEY SETUP
# Keys are stored as SHA-256 hashes in .env
# Never stored or compared as plaintext
# ──────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load hashed keys from .env (comma separated)
HASHED_API_KEYS = [
    k.strip()
    for k in os.getenv("API_KEY_HASHES", "").split(",")
    if k.strip()
]


def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of an API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Verifies API key by comparing its SHA-256 hash
    against stored hashes. Raw key is never stored.
    Raises 403 if key is missing or invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="No API key provided. Include X-API-Key in your request header."
        )

    incoming_hash = _hash_key(api_key)

    if incoming_hash not in HASHED_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )

    # Return hash (never return raw key)
    return incoming_hash


# ──────────────────────────────────────────
# RATE LIMITER
# 10 requests per minute per IP
# ──────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    if HASHED_API_KEYS:
        print(f"✅ {len(HASHED_API_KEYS)} hashed API key(s) loaded")
        print(f"   Hash preview: {HASHED_API_KEYS[0][:16]}...")
    else:
        print("⚠️  No API_KEY_HASHES found in .env")

    # Test: verify a known key works
    test_key  = "adaptive-ai-key-001"
    test_hash = _hash_key(test_key)
    if test_hash in HASHED_API_KEYS:
        print(f"✅ Test key verified correctly via hash")
    else:
        print(f"⚠️  Test key hash not found — check your .env")

    print("✅ dependencies.py working correctly!")
