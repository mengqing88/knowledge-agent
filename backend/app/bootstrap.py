"""Explicit local bootstrap; random password is stored locally, never in source control."""
import argparse
import os
import secrets
from pathlib import Path

from sqlalchemy import select

from app.core.auth import hash_password
from app.core.config import ROOT
from app.core.db import Session
from app.core.models import Corpus, Membership, User


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="admin")
    args = parser.parse_args()
    with Session.begin() as db:
        user = db.scalar(select(User).where(User.username == args.username))
        if user:
            print("管理员账号已存在，保留原密码。")
            return
        password = secrets.token_urlsafe(15)
        user = User(id="local-user", username=args.username, password_hash=hash_password(password))
        db.add(user)
        db.flush()
        for corpus in db.scalars(select(Corpus).where(Corpus.owner_id == user.id)):
            db.add(Membership(corpus_id=corpus.id, user_id=user.id, role="admin"))
    path = Path(os.environ.get("KB_RUNTIME_DIR", str(ROOT / "data"))) / "admin-login.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"账号: {args.username}\n密码: {password}\n本机初始化生成；请勿提交或公开分享此文件。\n", encoding="utf-8")
    print(f"已创建管理员，登录信息保存在 {path}")
