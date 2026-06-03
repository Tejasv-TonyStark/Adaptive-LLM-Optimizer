# auth/user_store.py

from dataclasses import dataclass

from auth.auth_handler import hash_password


@dataclass
class AuthUser:
    id: int
    username: str
    email: str
    hashed_password: str
    is_active: bool = True


_users_by_id: dict[int, AuthUser] = {}
_users_by_username: dict[str, AuthUser] = {}
_users_by_email: dict[str, AuthUser] = {}
_next_user_id = 1


def create_user(username: str, email: str, password: str) -> AuthUser:
    global _next_user_id

    user = AuthUser(
        id=_next_user_id,
        username=username,
        email=email,
        hashed_password=hash_password(password),
    )
    _next_user_id += 1

    _users_by_id[user.id] = user
    _users_by_username[user.username.lower()] = user
    _users_by_email[user.email.lower()] = user
    return user


def get_user_by_id(user_id: int) -> AuthUser | None:
    return _users_by_id.get(user_id)


def get_user_by_username(username: str) -> AuthUser | None:
    return _users_by_username.get(username.lower())


def get_user_by_email(email: str) -> AuthUser | None:
    return _users_by_email.get(email.lower())


def seed_demo_user() -> None:
    if get_user_by_username("tejasv"):
        return
    create_user(
        username="tejasv",
        email="tejasv@example.com",
        password="mypassword123",
    )


seed_demo_user()
