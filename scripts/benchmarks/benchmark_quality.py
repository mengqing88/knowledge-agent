"""Run fixed fictional answer ablations in a disposable PostgreSQL/Milvus namespace.

The default smoke run has three questions and two arms (at most 36 chat calls).
Full-suite execution must be selected explicitly. Semantic metrics stay pending
until independent human review; no application setting or production data changes.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.agents.governance import usage_summary  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.models import (  # noqa: E402
    Corpus,
    ModelCall,
    Paper,
    PaperVersion,
    ParseRevision,
    QuestionRun,
    RetrievalRecord,
    RunEvent,
    User,
)
from app.core.schemas import RunCreate  # noqa: E402
from app.integrations.providers import ModelProvider  # noqa: E402
from app.jobs.services import claim_job, create_run, create_snapshot, run_data  # noqa: E402
from app.jobs.worker import process_job  # noqa: E402
from app.knowledge.chunking import ChunkPolicy  # noqa: E402
from sqlalchemy import select  # noqa: E402

from evaluation.quality_profiles import LIMITS, PROFILES  # noqa: E402
from evaluation.report import summarize  # noqa: E402
from evaluation.review import make_template, summarize_manual  # noqa: E402
from scripts.benchmarks.benchmark_chunking import CachedEmbedding, evaluate_strategy, load_corpus, scratch  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_suite(path):
    suite = json.loads(path.read_text(encoding="utf-8"))
    corpus_path = ROOT / suite["corpus_manifest"]
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != suite["corpus_manifest_sha256"]:
        raise ValueError("Fixed corpus manifest changed; explicitly version the suite before running")
    _, documents, questions, blocks = load_corpus(corpus_path)
    by_id = {q["id"]: q for q in questions}
    cases = []
    for case in suite["questions"]:
        if case.get("retrieval_question_id"):
            reference = by_id[case["retrieval_question_id"]]
            if case["question"] != reference["question"]:
                raise ValueError("Frozen retrieval question differs")
        cases.append(case)
    if not cases or len({q["question_id"] for q in cases}) != len(cases):
        raise ValueError("Empty or duplicate questions")
    return suite, documents, questions, blocks


def trace_data(db, run):
    calls = [{"role": c.role, "model": c.model, "status": c.status, "usage_known": c.usage_known,
              "input_tokens": c.input_tokens if c.usage_known else None,
              "output_tokens": c.output_tokens if c.usage_known else None,
              "elapsed_seconds": c.elapsed_seconds, "request": c.request, "response": c.response}
             for c in db.scalars(select(ModelCall).where(ModelCall.run_id == run.id).order_by(ModelCall.created_at))]
    retrievals = [{"attempt": r.attempt, "round": r.round, "query": r.query, "rankings": r.rankings}
                  for r in db.scalars(select(RetrievalRecord).where(RetrievalRecord.run_id == run.id).order_by(
                      RetrievalRecord.created_at))]
    events = [{"kind": e.kind, "attempt": e.attempt, "payload": e.payload}
              for e in db.scalars(select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.seq))]
    return {"model_calls": calls, "retrievals": retrievals, "events": events,
            "accounting": usage_summary(db, run.id, run.config)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "evaluation/quality_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/rag-improvements-validation/quality-runs")
    parser.add_argument("--full-suite", action="store_true", help="Explicitly run every question and all four arms")
    args = parser.parse_args()
    suite, documents, retrieval_questions, blocks = load_suite(args.suite)
    questions = suite["questions"] if args.full_suite else [
        q for q in suite["questions"] if q["question_id"] in suite["smoke_question_ids"]]
    profiles = list(PROFILES) if args.full_suite else ["without_evidence_gate", "full_agent"]
    settings = Settings(**LIMITS)
    settings = settings.model_copy(update={"chunk_strategy": "fixed", "chunk_size_tokens": 512,
                                           "chunk_overlap_tokens": 64})
    output = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6])
    output.mkdir(parents=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "dev_only": True,
                "fictional": True, "human_reviewed": False, "manual_semantic_review": "pending",
                "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
                "corpus_manifest_sha256": suite["corpus_manifest_sha256"], "mode_order": profiles,
                "profiles": {p: PROFILES[p] for p in profiles}, "limits": LIMITS,
                "questions": questions, "status": "preparing", "execution": "synchronous isolated worker, no Celery/checkpoints"}
    write(output / "manifest.json", manifest)
    print(output.resolve(), flush=True)
    provider = CachedEmbedding(settings, max_tokens=100000, max_texts=200,
                               cache_dir=ROOT / "tmp/chunking-embedding-cache")
    try:
        with scratch(settings) as (sessions, store, isolated):
            manifest.update(scratch_database=sessions.kw["bind"].url.database,
                            scratch_prefix=isolated.milvus_collection_prefix)
            write(output / "manifest.json", manifest)
            with sessions.begin() as db:
                db.add(User(id="benchmark", username="benchmark", password_hash="disabled-experiment-account"))
                db.flush()
                db.add(Corpus(id="benchmark", owner_id="benchmark", name="Fictional fixed development evaluation"))
                db.flush()
                for key, (entry, _, _) in documents.items():
                    db.add(Paper(id=key, corpus_id="benchmark", title=key))
                    db.flush()
                    db.add(PaperVersion(id=key, paper_id=key, filename=entry["filename"], storage_key=entry["filename"],
                                        sha256=entry["sha256"], byte_size=1, status="ready", is_active=True))
                    db.flush()
                    db.add(ParseRevision(id=key, paper_version_id=key, parser="frozen-fixture",
                                         config_version="quality-experiment-v1", status="ready"))
                db.flush()
                db.add_all(blocks)
            evaluate_strategy(sessions, store, isolated, provider, ChunkPolicy(), documents, retrieval_questions)
            with sessions() as db:
                snapshot = create_snapshot(db, "benchmark", list(documents), "benchmark")
                manifest.update(snapshot={"id": snapshot.id, "manifest": snapshot.manifest}, status="running")
            write(output / "manifest.json", manifest)
            for profile in profiles:
                directory = output / profile
                directory.mkdir()
            # Interleave the arms to reduce confounding by server load over time.
            for number, question in enumerate(questions, 1):
                for profile in profiles:
                    config = PROFILES[profile]
                    arm = isolated.model_copy(update={"evidence_gate_enabled": config["evidence_gate_enabled"]})
                    started = time.monotonic()
                    row = {"question_id": question["question_id"], "mode": profile}
                    with sessions() as db:
                        run = create_run(db, RunCreate(snapshot_id=snapshot.id, question=question["question"],
                                         mode=config["mode"]), uuid4().hex, "benchmark", arm)
                        row["run_id"], row["execution_manifest"] = run.id, run.config
                    process_job(sessions, arm, ModelProvider(arm), claim_job(sessions, arm))
                    with sessions() as db:
                        run = db.get(QuestionRun, row["run_id"])
                        row.update(run=run_data(db, run, "benchmark"), trace=trace_data(db, run))
                    row["wall_seconds"] = round(time.monotonic() - started, 3)
                    write(output / profile / f"{number:04d}.json", row)
                    print(profile, question["question_id"], row["run"]["status"], row["wall_seconds"], flush=True)
        manifest.update(status="completed", cleanup="passed", embedding=provider.summary())
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, embedding=provider.summary())
        raise
    finally:
        write(output / "manifest.json", manifest)
    summary = {p: summarize(output / p) for p in profiles}
    summary["manual_metrics"] = {p: summarize_manual(output, p) for p in profiles}
    write(output / "summary.json", summary)
    write(output / "reviews.pending.json", make_template(output))
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
