"""
Password hashing (Argon2 through pwdlib) and JWT access tokens (PyJWT, HS256 unless
JWT_ALGORITHM picks HS384 or HS512).

Routes get the logged-in user with a `user: CurrentUser` parameter.
"""

import logging
import math
import secrets
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from sqlalchemy.orm import Session

from config import settings
from database import DbSession
from models import User
from schemas import MAX_ID, USERNAME

log = logging.getLogger("uvicorn.error")

ALGORITHM = settings.jwt_algorithm

if settings.jwt_secret_key:
    SECRET_KEY = settings.jwt_secret_key
else:
    # 64 bytes is long enough for every JWT_ALGORITHM config.py allows
    SECRET_KEY = secrets.token_urlsafe(64)
    log.warning("JWT_SECRET_KEY is not set, using a random one. Tokens stop working when the API restarts")

password_hash = PasswordHash.recommended()
# compared against when the username doesn't exist, so an unknown user takes
# as long to reject as a wrong password and the timing doesn't leak which it was
DUMMY_HASH = password_hash.hash("not-a-real-password")

# tokenUrl is what Swagger's Authorize button posts the form to
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def hash_password(password: str) -> str:
    """Argon2 hash of a password, with its salt and parameters in the string."""
    return password_hash.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Whether a password matches a stored hash."""
    return password_hash.verify(plain, hashed)


def authenticate(db: Session, username: str, password: str) -> User | None:
    """The user these credentials belong to, or None for an unknown user or a wrong password."""
    username = username.strip().lower()
    # a name that could never have been registered (a NUL byte in it was a 500
    # from Postgres) is just an unknown user, without asking the database
    user = None
    if USERNAME.match(username):
        user = db.query(User).filter(User.username == username).one_or_none()
    if user is None:
        verify_password(password, DUMMY_HASH)
        return None
    return user if verify_password(password, user.hashed_password) else None


# ── Failed sign-ins ────────────────────────────────────────────────────────

# after 5 wrong passwords for one username within 15 minutes, that username
# can't sign in until the oldest of them is 15 minutes old. Each attempt is
# counted when it starts and forgotten if the password was right. Counting it
# after the password check let 19 of 40 guesses sent at once through, because
# they all passed the check before any was counted. Attempts while it's
# blocked aren't counted, so it always opens again. Keyed on the username, not
# the address: everything through the UI comes from the frontend container.
# Kept in memory, which works because the API runs one uvicorn worker, and a
# restart clears it
MAX_FAILED_LOGINS = 5
FAILED_LOGIN_WINDOW = 15 * 60
# drop usernames whose failures have all expired once this many are tracked
FAILED_LOGIN_KEYS = 10_000
# monotonic, so changing the system clock can't open or extend a block. Tests replace it
clock = time.monotonic
failed_logins: dict[str, list[float]] = {}
failed_logins_lock = threading.Lock()


def login_key(username: str) -> str:
    """The username as authenticate() looks it up, capped so a huge one can't take up memory."""
    return username.strip().lower()[:100]


def start_login_attempt(username: str) -> int:
    """
    Count a sign-in attempt for this username before its password is checked.
    Returns 0 if it can go ahead, otherwise the seconds until one can, and
    then it isn't counted. Wrong passwords and unknown usernames stay counted.
    """
    key, now = login_key(username), clock()
    with failed_logins_lock:
        recent = [t for t in failed_logins.get(key, []) if now - t < FAILED_LOGIN_WINDOW]
        if len(recent) >= MAX_FAILED_LOGINS:
            failed_logins[key] = recent
            return math.ceil(FAILED_LOGIN_WINDOW - (now - recent[-MAX_FAILED_LOGINS]))
        if key not in failed_logins and len(failed_logins) >= FAILED_LOGIN_KEYS:
            for stale in [k for k, times in failed_logins.items() if now - times[-1] >= FAILED_LOGIN_WINDOW]:
                del failed_logins[stale]
        failed_logins[key] = [*recent, now]
        return 0


def clear_failed_logins(username: str):
    """Forget a username's attempts once it signs in."""
    with failed_logins_lock:
        failed_logins.pop(login_key(username), None)


# ── Tokens ─────────────────────────────────────────────────────────────────


def create_access_token(user_id: int) -> str:
    """A signed JWT for the user, with `sub`, `iat` and `exp` claims."""
    now = datetime.now(UTC)
    claims = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=settings.jwt_expire_minutes)}
    return jwt.encode(claims, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: DbSession) -> User:
    """The user a bearer token was issued to. Any token that isn't valid for a current account is a 401."""
    try:
        # algorithms is a list on purpose: never accept whatever the token header claims
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"require": ["exp", "iat", "sub"]})
        user_id = int(payload["sub"])
        issued_at = int(payload["iat"])
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise CREDENTIALS_ERROR from None
    # ids are Postgres INTEGER, a bigger one made db.get raise instead of finding nothing
    if not 1 <= user_id <= MAX_ID:
        raise CREDENTIALS_ERROR
    # a valid token for a deleted account
    user = db.get(User, user_id)
    if user is None:
        raise CREDENTIALS_ERROR
    # a token from before this account existed was issued to whoever had the
    # id before: `docker-compose down -v` starts Postgres ids at 1 again, and
    # old tokens survive that when JWT_SECRET_KEY is set. iat is whole
    # seconds, so the account's creation time is rounded down to match.
    # SQLite returns it without the timezone, but it was saved in UTC
    created_at = user.created_at if user.created_at.tzinfo else user.created_at.replace(tzinfo=UTC)
    if issued_at < int(created_at.timestamp()):
        raise CREDENTIALS_ERROR
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
