"""
Password hashing (Argon2 through pwdlib) and JWT access tokens (PyJWT, HS256).

Routes get the logged-in user with a `user: CurrentUser` parameter.
"""

import logging
import secrets
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

ALGORITHM = "HS256"

if settings.jwt_secret:
    SECRET_KEY = settings.jwt_secret
else:
    SECRET_KEY = secrets.token_urlsafe(32)
    log.warning("JWT_SECRET is not set, using a random one. Tokens stop working when the API restarts")

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
    return password_hash.hash(password)


def authenticate(db: Session, username: str, password: str) -> User | None:
    username = username.strip().lower()
    # a name that could never have been registered (a NUL byte in it was a 500
    # from Postgres) is just an unknown user, without asking the database
    user = None
    if USERNAME.match(username):
        user = db.query(User).filter(User.username == username).one_or_none()
    if user is None:
        password_hash.verify(password, DUMMY_HASH)
        return None
    return user if password_hash.verify(password, user.hashed_password) else None


def create_access_token(user_id: int) -> str:
    now = datetime.now(UTC)
    claims = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=settings.token_minutes)}
    return jwt.encode(claims, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: DbSession) -> User:
    try:
        # algorithms is a list on purpose: never accept whatever the token header claims
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"require": ["exp", "sub"]})
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise CREDENTIALS_ERROR from None
    # ids are Postgres INTEGER, a bigger one made db.get raise instead of finding nothing
    if not 1 <= user_id <= MAX_ID:
        raise CREDENTIALS_ERROR
    # a valid token for a deleted account
    user = db.get(User, user_id)
    if user is None:
        raise CREDENTIALS_ERROR
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
