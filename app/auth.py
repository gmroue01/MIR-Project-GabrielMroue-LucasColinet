"""
Simple single-password JWT authentication.

Set APP_PASSWORD env var to enable. Leave unset for open access (dev).
Set JWT_SECRET to a random string in production:
    python -c "import secrets; print(secrets.token_hex(32))"
"""
import hmac
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
SECRET_KEY   = os.environ.get("JWT_SECRET", "dev-secret-not-for-production")
ALGORITHM    = "HS256"
EXPIRE_HOURS = 24

# Fail fast: if auth is enabled with the default secret, tokens are forgeable.
if APP_PASSWORD and SECRET_KEY == "dev-secret-not-for-production":
    raise RuntimeError(
        "JWT_SECRET must be set to a strong random value when APP_PASSWORD is configured. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

# Rate limiting: max 5 failed attempts per IP within 5 minutes
_MAX_FAILURES  = 5
_WINDOW_S      = 300          # 5-minute sliding window
_failed: dict  = defaultdict(list)   # ip -> [failure timestamps]


def check_rate_limit(ip: str) -> bool:
    now     = time.monotonic()
    cutoff  = now - _WINDOW_S
    recent  = [t for t in _failed[ip] if t > cutoff]
    _failed[ip] = recent
    return len(recent) < _MAX_FAILURES


def record_failure(ip: str) -> None:
    _failed[ip].append(time.monotonic())


def auth_enabled() -> bool:
    return bool(APP_PASSWORD)


def verify_password(password: str) -> bool:
    if not APP_PASSWORD:
        return True
    return hmac.compare_digest(password.encode(), APP_PASSWORD.encode())


def create_token() -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=EXPIRE_HOURS)
    return jwt.encode({"exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> bool:
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return True
    except JWTError:
        return False
