import os
from fastapi import Depends, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session
from auth.auth_handler import verify_jwt
from database.connection import get_db
from database.models import User

limiter = Limiter(key_func=get_remote_address, storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"))
def current_user(token=Depends(verify_jwt), db: Session=Depends(get_db)):
    user = db.get(User, int(token["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(401, "User no longer exists or is inactive.")
    return user
def admin_user(user=Depends(current_user)):
    admins = {name.strip().casefold() for name in os.getenv("ADMIN_USERNAMES", "").split(",") if name.strip()}
    if user.username.casefold() not in admins:
        raise HTTPException(403, "Administrator access required.")
    return user
