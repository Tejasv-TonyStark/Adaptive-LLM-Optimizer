from datetime import datetime, timedelta, timezone
import os
import bcrypt
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
load_dotenv()
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24
def signing_key():
    key = os.getenv("JWT_SECRET_KEY", "")
    if len(key.encode()) < 32 or key == "fallback-secret-change-this":
        raise RuntimeError("JWT_SECRET_KEY must contain at least 32 bytes; generate a random secret.")
    return key
def hash_password(plain):
    raw = plain.encode("utf-8")
    if len(raw) > 72:
        raise ValueError("Password exceeds bcrypt's 72-byte limit")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode()
def verify_password(plain, hashed):
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except (TypeError, ValueError):
        return False
def create_token(user_id, username):
    now = datetime.now(timezone.utc)
    return jwt.encode(dict(sub=str(user_id), username=username, iat=now,
                           exp=now+timedelta(hours=TOKEN_EXPIRE_HOURS)),
                      signing_key(), algorithm=ALGORITHM)
def decode_token(token):
    try:
        data = jwt.decode(token, signing_key(), algorithms=[ALGORITHM],
                          options={"require_exp": True, "require_iat": True, "require_sub": True})
        if not isinstance(data["sub"], str) or not data["sub"].isdigit() or int(data["sub"]) < 1:
            raise JWTError("Invalid subject")
        return data
    except JWTError:
        raise HTTPException(401, "Invalid or expired token.")
bearer_scheme = HTTPBearer(auto_error=False)
def verify_jwt(credentials: HTTPAuthorizationCredentials=Security(bearer_scheme)):
    if not credentials:
        raise HTTPException(401, "Authentication required.")
    return decode_token(credentials.credentials)
