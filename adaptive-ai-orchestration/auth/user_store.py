"""Persistent users; no automatic accounts or process-local identity state."""
from sqlalchemy import func, or_
from database.models import User
from auth.auth_handler import hash_password
def get_user_by_username(db, username):
    return db.query(User).filter(func.lower(User.username) == username.strip().casefold()).first()
def get_user_by_email(db, email):
    return db.query(User).filter(func.lower(User.email) == email.strip().casefold()).first()
def create_user(db, username, email, password):
    user = User(username=username.strip().casefold(), email=email.strip().casefold(),
                hashed_password=hash_password(password), is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
