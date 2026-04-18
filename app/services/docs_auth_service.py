from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from jose import JWTError, jwt

from app.config import settings


DOCS_TOKEN_AUDIENCE = "swagger-docs"
DOCS_TOKEN_SUBJECT = "swagger-access"
DOCS_ALGORITHM = "HS256"


def issue_docs_token(requested_by: str | None = None) -> dict:
    if not settings.docs_jwt_secret:
        raise RuntimeError("Missing DOCS_JWT_SECRET. Configure it before issuing docs tokens.")

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=settings.docs_token_expiry_hours)
    token = jwt.encode(
        {
            "sub": DOCS_TOKEN_SUBJECT,
            "aud": DOCS_TOKEN_AUDIENCE,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "requested_by": requested_by or "manual",
        },
        settings.docs_jwt_secret,
        algorithm=DOCS_ALGORITHM,
    )
    return {
        "status": "success",
        "access_token": token,
        "token_type": "bearer",
        "expires_in_seconds": settings.docs_token_expiry_hours * 3600,
        "docs_url": f"{settings.public_base_url.rstrip('/')}/docs?token={token}",
    }


def verify_docs_request(request: Request) -> str:
    token = request.query_params.get("token") or _bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Swagger access token required.")
    return verify_docs_token(token)


def verify_docs_token(token: str) -> str:
    if not settings.docs_jwt_secret:
        raise HTTPException(status_code=503, detail="Swagger auth is not configured.")
    try:
        payload = jwt.decode(
            token,
            settings.docs_jwt_secret,
            algorithms=[DOCS_ALGORITHM],
            audience=DOCS_TOKEN_AUDIENCE,
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired Swagger access token.")

    if payload.get("sub") != DOCS_TOKEN_SUBJECT:
        raise HTTPException(status_code=401, detail="Invalid Swagger access token subject.")

    return str(payload.get("requested_by") or "manual")


def require_docs_admin(request: Request) -> None:
    if not settings.docs_admin_secret:
        raise HTTPException(status_code=503, detail="Swagger admin secret is not configured.")

    supplied = request.headers.get("X-Docs-Admin-Secret", "").strip()
    if supplied != settings.docs_admin_secret:
        raise HTTPException(status_code=401, detail="Invalid docs admin secret.")


def _bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "").strip()
    if not authorization.lower().startswith("bearer "):
        return None
    return authorization.split(" ", 1)[1].strip() or None
