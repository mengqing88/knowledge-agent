import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import text

from .config import get_settings
from .db import Session
from .errors import DomainError, ProviderError
from .providers import ModelProvider
from .vector_store import MilvusVectorStore


class Probe(BaseModel):
    status: Literal["ok"]


def doctor(output: str | None):
    settings = get_settings()
    provider = ModelProvider(settings)
    result = {"chat_model": settings.model, "embedding_model": settings.embedding_model}
    try:
        _, usage = provider.structured("connection_probe", "连接测试。仅返回 {\"status\":\"ok\"}。", {}, Probe)
        result["chat"] = {"status": "ok", **usage}
    except ProviderError as exc:
        result["chat"] = {"status": "failed", "error": str(exc)}
    try:
        vector = provider.embed(["Connection probe: paper evidence retrieval."])[0]
        result["embedding"] = {"status": "ok", "dimensions": len(vector)}
    except ProviderError as exc:
        result["embedding"] = {"status": "failed", "error": str(exc)}
    try:
        with Session() as db:
            db.execute(text("SELECT 1"))
            result["database"] = {"status": "ok", "backend": db.bind.dialect.name}
    except Exception:
        result["database"] = {"status": "failed", "error": "业务数据库不可用"}
    store = None
    try:
        store = MilvusVectorStore(settings)
        result["vector_store"] = store.health()
    except ProviderError as exc:
        result["vector_store"] = {"status": "failed", "error": str(exc)}
    finally:
        if store is not None:
            store.close()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    return 0 if all(result[k]["status"] == "ok" for k in
                    ("chat", "embedding", "database", "vector_store")) else 1


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Paper QA maintenance commands")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor", help="Check real chat, embedding, database and Milvus connections; never prints keys")
    check.add_argument("--output")
    transfer = sub.add_parser("migrate-vectors", help="Copy and verify legacy PostgreSQL vectors in Milvus")
    transfer.add_argument("--index-id", action="append")
    transfer.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor(args.output)
    if args.command == "migrate-vectors":
        from .vector_migration import migrate_vectors

        with Session() as db:
            if db.bind.dialect.name != "postgresql":
                parser.error("migrate-vectors 只操作 PostgreSQL；SQLite 演示数据请保留原状")
        store = None
        try:
            settings = get_settings()
            store = MilvusVectorStore(settings)
            result = migrate_vectors(Session, settings, store, index_ids=args.index_id,
                                     batch_size=args.batch_size)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (DomainError, ProviderError) as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
            return 1
        finally:
            if store is not None:
                store.close()


if __name__ == "__main__":
    raise SystemExit(main())
