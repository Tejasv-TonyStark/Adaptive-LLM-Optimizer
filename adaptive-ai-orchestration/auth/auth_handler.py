# auth/auth_handler.py

from datetime import datetime, timedelta, timezone
import os

import bcrypt
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

load_dotenv()

# CONFIG
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback-secret-change-this")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


# PASSWORD HASHING - bcrypt
def hash_password(plain: str) -> str:
    password_bytes = plain.encode("utf-8")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (TypeError, ValueError):
        return False


# JWT TOKEN
def create_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": now + timedelta(hours=TOKEN_EXPIRE_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again.",
        )


# BEARER SCHEME
bearer_scheme = HTTPBearer(auto_error=False)


def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")
    return decode_token(credentials.credentials)


# TEST
if __name__ == "__main__":
    print("\n-- Auth Handler Test --\n")

    raw = "mypassword123"
    hashed = hash_password(raw)
    print(f"[OK] Password hash:    {hashed[:30]}...")
    print(f"[OK] Correct verify:   {verify_password(raw, hashed)}")
    print(f"[OK] Wrong verify:     {verify_password('wrongpass', hashed)}")

    token = create_token(user_id=1, username="tejasv")
    payload = decode_token(token)
    print(f"\n[OK] Token created:    {token[:40]}...")
    print(f"[OK] Decoded user:     {payload['username']} (id={payload['sub']})")

    try:
        decode_token("invalid.token.here")
    except Exception as exc:
        print(f"[OK] Invalid rejected: {exc.detail}")

    print("\n[OK] auth_handler.py working correctly!")
