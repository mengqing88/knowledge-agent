"""One isolated, interleaved 12 x 2 real-model evaluation, with a durable CNY 3 ledger.

Requires a uniquely named deploy/compose.acceptance.yaml stack. Never uses the
application database/volumes, never edits .env, never retries an experiment.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.config import Settings  # noqa: E402
from app.models import Corpus, Paper, PaperVersion, QuestionRun, User  # noqa: E402
from app.schemas import RunCreate  # noqa: E402
from app.services import claim_job, create_run, create_snapshot, run_data  # noqa: E402
from app.worker import process_job  # noqa: E402

from evaluation.budget import BudgetLedger, BudgetProvider  # noqa: E402
from evaluation.quality_profiles import LIMITS  # noqa: E402
from evaluation.report import summarize  # noqa: E402
from evaluation.review import make_template, summarize_agent, summarize_manual  # noqa: E402
from evaluation.run import case_order  # noqa: E402
from scripts.benchmark_chunking import scratch  # noqa: E402
from scripts.benchmark_quality import trace_data, write  # noqa: E402
from scripts.isolated_acceptance import port  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/public-3yuan-v1")
    parser.add_argument("--prices", type=Path, default=ROOT / "evaluation/public_docs_v1/prices.json")
    args = parser.parse_args()
    if not args.project.startswith("kbaccept-"):
        raise SystemExit("Use a dedicated acceptance Compose project")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise SystemExit("This experiment already exists. No reruns or budget resets.")
    prices = json.loads(args.prices.read_text(encoding="utf-8"))
    for evidence in prices["evidence"]:
        if digest(ROOT / evidence["file"]) != evidence["sha256"]:
            raise SystemExit("Price evidence hash changed")
    ledger = BudgetLedger(output / "cost-ledger.json", prices["rates"])
    source = ROOT / "evaluation/public_docs_v1"
    suite = json.loads((source / "suite.json").read_text(encoding="utf-8"))
    corpus = json.loads((source / "corpus.json").read_text(encoding="utf-8"))
    if digest(source / "corpus.json") != suite["corpus_manifest_sha256"]:
        raise SystemExit("Corpus manifest changed")
    for doc in corpus["documents"]:
        if digest(source / doc["filename"]) != doc["sha256"]:
            raise SystemExit("Corpus excerpt changed")
        if digest(source / "originals" / doc["source"]["file"]) != doc["source"]["sha256"]:
            raise SystemExit("Original source changed")
    modes = ["baseline_rag", "full_agent"]
    settings = Settings(**{**LIMITS, "max_context_chars": 8000, "max_run_tokens": 180000,
        "query_rewrite_enabled": False, "query_max_per_subquestion": 2,
        "provider_quota_enabled": False,
        "database_url": f"postgresql+psycopg://acceptance:acceptance-local-only@127.0.0.1:{port(args.project, 'db', 5432)}/acceptance",
        "milvus_uri": f"http://127.0.0.1:{port(args.project, 'milvus', 19530)}",
        "milvus_token": "", "milvus_db_name": "default",
        "storage_dir": output / "documents",
        "chat_input_price_per_million": float(prices["rates"]["chat"]["input"]),
        "chat_output_price_per_million": float(prices["rates"]["chat"]["output"]),
        "embedding_price_per_million": float(prices["rates"]["embedding"]["input"]),
        "price_currency": "CNY", "price_source": "evaluation/public_docs_v1/prices.json"})
    settings.storage_dir.mkdir(exist_ok=True)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), dev_only=True, fictional=False,
                    human_reviewed=False, review_source="agent", manual_semantic_review="pending",
                    question_set_sha256=digest(source / "suite.json"),
                    corpus_manifest_sha256=digest(source / "corpus.json"),
                    questions=suite["questions"], mode_order=modes, limits=LIMITS,
                    model=settings.model, embedding_model=settings.embedding_model,
                    project=args.project, status="preparing", execution="synchronous isolated real worker; shared vector cache",
                    price_manifest_sha256=digest(args.prices))
    write(output / "manifest.json", manifest)
    for mode in modes:
        (output / mode).mkdir()
    started = time.monotonic()
    cache = {}
    stop_reason = None
    try:
        with scratch(settings) as (sessions, _store, isolated):
            with sessions.begin() as db:
                db.add(User(id="benchmark", username="benchmark", password_hash="disabled-experiment-account"))
                db.flush()
                db.add(Corpus(id="benchmark", owner_id="benchmark", name="Official public excerpts"))
                db.flush()
                for doc in corpus["documents"]:
                    data = (source / doc["filename"]).read_bytes()
                    (settings.storage_dir / doc["filename"]).write_bytes(data)
                    db.add(Paper(id=doc["id"], corpus_id="benchmark", title=doc["title"]))
                    db.flush()
                    db.add(PaperVersion(id=doc["id"], paper_id=doc["id"], filename=doc["filename"],
                           storage_key=doc["filename"], sha256=doc["sha256"], byte_size=len(data), status="uploaded"))
            for doc in corpus["documents"]:
                ledger.guard()
                provider = BudgetProvider(isolated, ledger, cache)
                process_job(sessions, isolated, provider, claim_job(sessions, isolated))
                with sessions() as db:
                    version = db.get(PaperVersion, doc["id"])
                    if version.status != "ready":
                        raise RuntimeError(f"Index failed: {doc['id']}: {version.error}")
            with sessions() as db:
                snapshot = create_snapshot(db, "benchmark", [d["id"] for d in corpus["documents"]], "benchmark")
                manifest.update(snapshot={"id": snapshot.id, "manifest": snapshot.manifest}, status="running")
                for question in manifest["questions"]:
                    question["snapshot_id"] = snapshot.id
            write(output / "manifest.json", manifest)
            for number, question, mode in case_order(suite["questions"], modes):
                ledger.guard()
                arm = isolated.model_copy(update={"evidence_gate_enabled": mode == "full_agent"})
                row = dict(question_id=question["question_id"], mode=mode)
                tick = time.monotonic()
                with sessions() as db:
                    run = create_run(db, RunCreate(snapshot_id=snapshot.id, question=question["question"], mode=mode),
                                     uuid4().hex, "benchmark", arm)
                    row.update(run_id=run.id, execution_manifest=run.config)
                process_job(sessions, arm, BudgetProvider(arm, ledger, cache), claim_job(sessions, arm))
                with sessions() as db:
                    run = db.get(QuestionRun, row["run_id"])
                    row.update(run=run_data(db, run, "benchmark"), trace=trace_data(db, run))
                row["wall_seconds"] = round(time.monotonic() - tick, 3)
                write(output / mode / f"{number:04d}.json", row)
                print(mode, question["question_id"], row["run"]["status"], f"CNY {ledger.total()}", flush=True)
    except Exception as exc:
        stop_reason = ledger.data.get("halt_reason") or type(exc).__name__
        manifest.update(error_type=type(exc).__name__)
        print("Stopped:", stop_reason, flush=True)
    finally:
        skipped = []
        for number, question, mode in case_order(suite["questions"], modes):
            path = output / mode / f"{number:04d}.json"
            if not path.exists():
                skipped.append([question["question_id"], mode])
                write(path, dict(question_id=question["question_id"], mode=mode,
                                 technical_error="not_executed", reason=stop_reason or "interrupted",
                                 run={"status": "not_executed"}))
        manifest.update(status="stopped" if skipped else "completed", stop_reason=stop_reason,
                        unexecuted=skipped, wall_seconds=round(time.monotonic() - started, 3),
                        estimated_cost_cny=str(ledger.total()), shared_unique_embeddings=len(cache),
                        cleanup="scratch context exited; disposable Compose stack owned by caller")
        write(output / "manifest.json", manifest)
        summary = {mode: summarize(output / mode) for mode in modes}
        summary["manual_metrics"] = {mode: summarize_manual(output, mode) for mode in modes}
        summary["agent_metrics"] = {mode: summarize_agent(output, mode) for mode in modes}
        write(output / "summary.json", summary)
        write(output / "agent-reviews.pending.json", make_template(output, "agent"))
    print(output, flush=True)


if __name__ == "__main__":
    main()
