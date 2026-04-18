import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from jose import JWTError, jwt

from app.config import settings
from app.models import UserResponse
from app.services.storage_service import create_user, get_user_by_email, get_user_by_id


AUTH_ALGORITHM = "HS256"


def signup_user(name: str, email: str, password: str) -> UserResponse:
    normalized_email = email.strip().lower()
    existing = get_user_by_email(normalized_email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    salt = os.urandom(16)
    password_hash = _hash_password(password, salt)
    user_id = create_user(name=name.strip(), email=normalized_email, password_hash=password_hash)
    user = get_user_by_id(user_id)
    return UserResponse(**user)


def login_user(email: str, password: str) -> UserResponse:
    normalized_email = email.strip().lower()
    user = get_user_by_email(normalized_email)
    if not user or not _verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return UserResponse(**user)


def issue_auth_token(user: UserResponse) -> str:
    secret = _jwt_secret()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=7)
    return jwt.encode(
        {
            "sub": str(user.id),
            "email": user.email,
            "iat": int(now.timestamp()),
            "exp": int(expires.timestamp()),
        },
        secret,
        algorithm=AUTH_ALGORITHM,
    )


def require_user_from_token(token: str) -> UserResponse:
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[AUTH_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    user_id = int(payload.get("sub", 0))
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return UserResponse(**user)


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return base64.b64encode(salt + digest).decode("utf-8")


def _verify_password(password: str, stored_hash: str) -> bool:
    raw = base64.b64decode(stored_hash.encode("utf-8"))
    salt, digest = raw[:16], raw[16:]
    fresh = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return hmac.compare_digest(digest, fresh)


def _jwt_secret() -> str:
    return settings.docs_jwt_secret or "deckmint-dev-auth-secret"
