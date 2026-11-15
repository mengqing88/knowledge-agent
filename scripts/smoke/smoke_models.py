"""Live-model smoke test with fictional PDFs and a temporary SQLite store.

This checks model integration and evidence mapping, NOT PostgreSQL/Milvus integration
or real-paper answer quality. Normal API/worker storage remains PostgreSQL.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.core.db import Base, make_engine  # noqa: E402
from app.core.models import Corpus, Paper, PaperVersion, QuestionRun, uid  # noqa: E402
from app.core.schemas import RunCreate  # noqa: E402
from app.integrations.providers import ModelProvider  # noqa: E402
from app.jobs.services import (  # noqa: E402
    claim_job,
    create_run,
    create_snapshot,
    run_data,
    source_hash_key,
)
from app.jobs.worker import process_job  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


def main():
    manifest = json.loads((ROOT / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="paperqa-live-") as temp:
        settings = get_settings().model_copy(update={
            "database_url": f"sqlite:///{Path(temp).as_posix()}/test.db",
            "storage_dir": ROOT / "fixtures", "max_run_seconds": 300,
        })
        provider = ModelProvider(settings)
        engine = make_engine(settings.database_url)
        Base.metadata.create_all(engine)
        sessions = sessionmaker(engine, expire_on_commit=False)
        corpus_id, version_ids = uid(), []
        with sessions.begin() as db:
            db.add(Corpus(id=corpus_id, owner_id="local-user", name="Fictional live smoke"))
            db.flush()
            for item in manifest["papers"]:
                paper_id, version_id = uid(), uid()
                content = (ROOT / "fixtures" / item["filename"]).read_bytes()
                sha, _ = source_hash_key(content)
                db.add(Paper(id=paper_id, corpus_id=corpus_id, title=item["title"]))
                db.flush()
                db.add(PaperVersion(id=version_id, paper_id=paper_id, sha256=sha,
                                    storage_key=item["filename"], filename=item["filename"],
                                    byte_size=len(content), status="uploaded"))
                version_ids.append(version_id)
        for _ in version_ids:
            job = claim_job(sessions, settings)
            print("Processing fictional PDF", flush=True)
            process_job(sessions, settings, provider, job)
        with sessions() as db:
            failures = [v.error for v in db.scalars(select(PaperVersion)) if v.status != "ready"]
            if failures:
                raise SystemExit("Ingestion failed: " + "; ".join(str(e) for e in failures))
            snapshot = create_snapshot(db, corpus_id, version_ids, "local-user")
            run = create_run(db, RunCreate(snapshot_id=snapshot.id, question=(
                "在这三篇虚构教学材料中，方法 M 是否全面优于基线？分别说明 F1、推理延迟和 GPU 显存的证据，保留数据集条件。"
            )), uid(), "local-user", settings)
            run_id = run.id
        print("Running real Planner / Retriever / Analyst / Verifier", flush=True)
        process_job(sessions, settings, provider, claim_job(sessions, settings))
        with sessions() as db:
            result = run_data(db, db.get(QuestionRun, run_id))
        result["validation_scope"] = {
            "fictional_materials": True, "live_chat_and_embedding": True,
            "database_backend": "temporary SQLite", "postgresql_milvus_verified": False,
            "formal_evaluation": False,
        }
        path = ROOT / "tmp/reports/live-model-smoke.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "error": result["error"],
                          "evidence_status": (result["answer"] or {}).get("evidence_status"),
                          "saved": str(path)}, ensure_ascii=False), flush=True)
        engine.dispose()
        return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
