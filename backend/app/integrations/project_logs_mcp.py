"""Actual project run logs over MCP; actor/corpus are injected by the server."""
import os
from datetime import timedelta
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from sqlalchemy import select, text

from app.core.access import require_role
from app.core.config import get_settings
from app.core.db import make_engine
from app.core.models import ModelCall, QuestionRun, RunEvent, Snapshot, now
from app.core.security import redact

mcp = FastMCP("Project runtime logs (read only)")


@mcp.tool()
def query_run_logs(run_id: Annotated[str | None, Field(max_length=40)] = None,
                   hours: Annotated[int, Field(ge=1, le=168)] = 24,
                   limit: Annotated[int, Field(ge=1, le=10)] = 5) -> dict:
    """查询本人的项目问答故障与调用状态，不返回他人记录、完整问题、提示词或模型正文。"""
    from sqlalchemy.orm import Session
    actor, corpus = os.environ["PROJECT_LOGS_ACTOR"], os.environ["PROJECT_LOGS_CORPUS"]
    settings = get_settings()
    engine = make_engine(os.environ["PROJECT_LOGS_DATABASE_URL"])
    try:
        with Session(engine) as db:
            if engine.dialect.name == "postgresql":
                db.execute(text("SET TRANSACTION READ ONLY"))
            else:
                db.execute(text("PRAGMA query_only=ON"))
            require_role(db, corpus, actor)
            query = select(QuestionRun).join(Snapshot).where(QuestionRun.owner_id == actor,
                Snapshot.corpus_id == corpus, QuestionRun.created_at >= now() - timedelta(hours=hours))
            if run_id:
                query = query.where(QuestionRun.id == run_id)
            rows = db.scalars(query.order_by(QuestionRun.created_at.desc()).limit(limit))
            records = []
            for run in rows:
                events = list(db.scalars(select(RunEvent).where(RunEvent.run_id == run.id,
                    RunEvent.kind.in_(["failed", "retrieval_degraded", "provider_http"])).order_by(
                    RunEvent.seq.desc()).limit(12)))
                calls = list(db.scalars(select(ModelCall).where(ModelCall.run_id == run.id).order_by(
                    ModelCall.created_at.desc()).limit(10)))
                records.append({"run_id": run.id, "status": run.status, "error": run.error,
                    "created_at": run.created_at.isoformat(),
                    "events": [{"kind": e.kind, "time": e.created_at.isoformat(),
                        "detail": {k: v for k, v in e.payload.items() if k in {
                            "code", "phase", "endpoint", "http_status", "http_attempt", "route"}}} for e in events],
                    "calls": [{"role": c.role, "model": c.model, "status": c.status,
                               "seconds": c.elapsed_seconds} for c in calls]})
            return redact({"source": "project_database_runtime_records", "observed_at": now().isoformat(),
                           "records": records, "note": "记录不证明故障根因；无记录不证明系统无故障。"}, settings)
    finally:
        engine.dispose()


if __name__ == "__main__":
    mcp.run(transport="stdio")
