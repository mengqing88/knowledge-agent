import argparse
import logging
import threading
import time
from contextlib import nullcontext

from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import select

from .config import get_settings
from .db import Session
from .errors import BudgetExhausted, DomainError, InterruptedRun, ProviderError
from .ingestion import ingest_version
from .models import Paper, PaperVersion, QuestionRun
from .providers import ModelProvider
from .security import redact
from .services import check_attempt, claim_job, emit, heartbeat
from .workflow import Workflow, initial_state

log = logging.getLogger("paperqa.worker")


def process_job(sessions, settings, provider, job, checkpointer=None):
    kind, object_id, attempt = job
    stopped = threading.Event()

    def keep_alive():
        while not stopped.wait(10):
            try:
                heartbeat(sessions, settings, job)
            except Exception:
                log.warning("任务租约续期失败；发布仍受 attempt 校验保护")

    keeper = threading.Thread(target=keep_alive, daemon=True)
    keeper.start()
    deadline = None

    def guard():
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExhausted("time" if kind == "run" else "ingestion_time")
        with sessions() as db:
            if kind == "run":
                check_attempt(db, object_id, attempt)
            else:
                version = db.get(PaperVersion, object_id)
                paper = db.get(Paper, version.paper_id) if version else None
                if (not version or version.attempt != attempt or
                        version.status not in ("parsing", "indexing") or not paper or paper.deleted):
                    raise InterruptedRun()

    def observe(payload):
        if kind == "run":
            with sessions.begin() as db:
                check_attempt(db, object_id, attempt, lock=True)
                emit(db, object_id, attempt, "provider_http", payload)

    try:
        with sessions() as db:
            seconds = (check_attempt(db, object_id, attempt).config["max_run_seconds"]
                       if kind == "run" else settings.max_ingest_seconds)
        deadline = time.monotonic() + seconds
        context = (provider.execution(deadline, guard, observe)
                   if hasattr(provider, "execution") else nullcontext())
        with context:
            guard()
            if kind == "ingest":
                ingest_version(sessions, settings, provider, object_id, attempt, guard)
            else:
                with sessions() as db:
                    state = initial_state(check_attempt(db, object_id, attempt))
                Workflow(sessions, settings, provider, object_id, attempt).execute(state, checkpointer)
    except InterruptedRun:
        # Cancellation/lease loss is already visible in persistent state.
        pass
    except Exception as exc:
        message = (str(exc) if isinstance(exc, (DomainError, ProviderError)) else
                   "文档入库达到运行时间上限" if isinstance(exc, BudgetExhausted) else
                   "内部执行错误，请检查服务器日志")
        message = redact(message, settings)
        if not isinstance(exc, (DomainError, ProviderError)):
            # Avoid including provider request/credential data in logs.
            log.error("执行失败：%s", type(exc).__name__)
        with sessions.begin() as db:
            model = QuestionRun if kind == "run" else PaperVersion
            obj = db.scalar(select(model).where(model.id == object_id).with_for_update())
            active = (obj and obj.attempt == attempt and
                      (obj.status == "running" and not obj.cancel_requested if kind == "run" else
                       obj.status in ("parsing", "indexing")))
            if active:
                obj.status, obj.error, obj.lease_until = "failed", message, None
                if kind == "run":
                    emit(db, obj.id, attempt, "failed", {"message": message,
                         "code": getattr(exc, "code", type(exc).__name__), "request_id": obj.id})
    finally:
        stopped.set()
        keeper.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description="Database-backed paper QA worker")
    parser.add_argument("--once", action="store_true", help="Process one queued job, then exit")
    parser.add_argument("--no-checkpoints", action="store_true", help="Diagnostics only; business state remains durable")
    args = parser.parse_args()
    settings = get_settings()
    if not settings.database_url.startswith("postgresql") and not args.no_checkpoints:
        raise SystemExit("正式 worker 需要 PostgreSQL；SQLite 仅能使用 --no-checkpoints 做本地演示")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dsn = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    cm = nullcontext(None) if args.no_checkpoints else PostgresSaver.from_conn_string(dsn)
    with cm as saver:
        if saver:
            saver.setup()
        log.info("Worker ready. Lease recovery starts a new attempt; old results cannot publish.")
        while True:
            job = claim_job(Session, settings)
            if job:
                process_job(Session, settings, ModelProvider(settings), job, saver)
            elif not args.once:
                time.sleep(1)
            if args.once:
                return


if __name__ == "__main__":
    main()
