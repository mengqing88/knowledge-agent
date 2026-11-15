import hashlib
import hmac
import secrets
from datetime import timedelta, timezone
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import select

from app.core.db import get_db
from app.core.errors import DomainError
from app.core.models import LoginSession, User, now


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return f"scrypt${salt}${digest}"


def verify_password(password, encoded):
    try:
        _, salt, expected = encoded.split("$")
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError):
        return False


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db, user, hours):
    token = secrets.token_urlsafe(32)
    db.add(LoginSession(user_id=user.id, token_hash=token_hash(token), expires_at=now() + timedelta(hours=hours)))
    db.commit()
    return token


def current_user(request: Request, db=Depends(get_db)):
    header = request.headers.get("authorization", "")
    token = header[7:] if header.startswith("Bearer ") else request.cookies.get("kb_session", "")
    session = db.scalar(select(LoginSession).where(LoginSession.token_hash == token_hash(token))) if token else None
    if not session or session.expires_at.replace(tzinfo=timezone.utc) <= now():
        raise DomainError("请先登录", 401)
    user = db.get(User, session.user_id)
    if not user:
        raise DomainError("账号不存在", 401)
    return user


CurrentUser = Annotated[User, Depends(current_user)]
