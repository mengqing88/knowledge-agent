"""One disposable PostgreSQL/Milvus/workflow smoke with real local BGE.

Uses the public synthetic test fixture and deterministic chat/query vectors.
No external model requests, existing data changes, or browser quality claims.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "backend/tests")]

import httpx  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.models import Chunk, IndexRevision, MCPInvocation, QuestionRun, RunEvent  # noqa: E402
from app.core.schemas import RunCreate  # noqa: E402
from app.integrations.provider_quota import client_for  # noqa: E402
from app.integrations.providers import ModelProvider  # noqa: E402
from app.jobs.services import claim_job, create_run  # noqa: E402
from app.jobs.worker import process_job  # noqa: E402
from app.knowledge.llama_rag import index_chunks  # noqa: E402
from app.knowledge.vector_store import collection_name  # noqa: E402
from conftest import FakeProvider, corpus  # noqa: E402
from sqlalchemy import select  # noqa: E402
from test_agent_upgrades import LogsProvider  # noqa: E402

from scripts.benchmarks.benchmark_chunking import scratch  # noqa: E402


def run(output, upgrades=False):
    if output.exists() or not output.resolve().is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("Use a new report under tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    settings = Settings().model_copy(update={"model": "test-model", "embedding_model": "test-embedding",
        "embedding_dimensions": 3, "query_rewrite_enabled": upgrades, "workflow_mcp_enabled": upgrades,
        "project_logs_mcp_enabled": upgrades, "provider_quota_enabled": upgrades,
        "provider_quota_namespace": "combined_check_" + uuid4().hex[:12],
        "retrieval_strategy": "facet_context", "max_run_seconds": 120,
        "answer_review_required": True, "role_model_routes": {}, "max_run_cost": None})
    result = {"status": "running", "scope": "one synthetic full_agent workflow, real PostgreSQL/Milvus/local BGE",
              "external_model_requests": 0, "new_external_cost_cny": 0, "independent_quality_evaluation": False}
    quota_client = client_for(settings.provider_quota_redis_url) if upgrades else None
    try:
        with scratch(settings) as (sessions, store, isolated):
            isolated.database_url = sessions.kw["bind"].url.render_as_string(hide_password=False)
            result.update(scratch_database=sessions.kw["bind"].url.database,
                          scratch_prefix=isolated.milvus_collection_prefix)
            corpus.__wrapped__(sessions, isolated)
            provider = FakeProvider()
            if upgrades:
                # Real Provider HTTP boundary and Redis reservation; transport
                # responds in memory so this cannot contact a paid service.
                gateway = ModelProvider(isolated)
                reservations = []
                gateway.observe = reservations.append
                transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"fixture": True}))
                http_client = httpx.Client
                class QuotaLogsProvider(LogsProvider):
                    def _probe(self):
                        with patch("app.integrations.providers.httpx.Client",
                                   lambda **kwargs: http_client(transport=transport, **kwargs)):
                            gateway._post("https://fixture.invalid", "local-test-key", "responses",
                                          {"model": "fixture", "max_output_tokens": 1})
                    def embed(self, *args):
                        self._probe()
                        return super().embed(*args)
                    def structured(self, *args):
                        self._probe()
                        return super().structured(*args)
                provider = QuotaLogsProvider()
                with sessions() as db:
                    create_run(db, RunCreate(snapshot_id="snap1", question="控制失败运行"),
                               "synthetic-failed-run", "local-user", isolated)
                process_job(sessions, isolated, FakeProvider(fail=True), claim_job(sessions, isolated))
            with sessions.begin() as db:
                index, chunk = db.get(IndexRevision, "index1"), db.get(Chunk, "chunk1")
                index.vector_collection = collection_name(index.id, isolated.milvus_collection_prefix)
                index_chunks(db, index, [chunk], [chunk.embedding], store, provider)
                chunk.embedding = None
            with sessions() as db:
                question = ("幂等性相关的最近问答失败日志怎样查看？该资料有没有测量延迟？" if upgrades else
                            "数据集 X 上 M 的 F1 是多少？该资料有没有测量延迟？")
                run_id = create_run(db, RunCreate(snapshot_id="snap1", question=question),
                                    "local-bge-runtime-smoke", "local-user", isolated).id
            process_job(sessions, isolated, provider, claim_job(sessions, isolated))
            with sessions() as db:
                run = db.get(QuestionRun, run_id)
                if run.status != "review_pending":
                    raise AssertionError(f"Expected review gate, got {run.status}: {run.error}")
                events = list(db.scalars(select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)))
                local = [e.payload for e in events if e.kind == "local_rerank_finished"]
                selection = [e.payload for e in events if e.kind == "context_selection"]
                if len(local) != 1 or len(selection) != 1 or selection[0]["facet_count"] != 2:
                    raise AssertionError("New selection pipeline was not exercised exactly once")
                if "Reranker" in provider.roles or run.config["retrieval_strategy"] != "facet_context":
                    raise AssertionError("Legacy ranking unexpectedly replaced the new selection")
                result.update(workflow_status=run.status, evidence_count=len(run.state["sources"]),
                              local_inference=local[0], selection=selection[0],
                              approval_gate="preserved", production_sql_vectors="not_used")
                if upgrades:
                    expansion = next(e.payload for e in events if e.kind == "lexical_expansion")
                    invocation = db.scalar(select(MCPInvocation).where(MCPInvocation.run_id == run_id))
                    reserved = [p for p in reservations if p.get("phase") == "quota_reserved"]
                    if expansion["query_count"] < 2 or invocation.status != "succeeded" or len(reserved) < 5:
                        raise AssertionError("Combined upgrades were not all exercised")
                    result["upgrades"] = {"lexical_query_count": expansion["query_count"],
                        "mcp_stdio_status": invocation.status, "quota_reservations": len(reserved),
                        "quota_shared_pools": len({r["pool"] for r in reserved}),
                        "provider_transport": "in_memory_httpx_mock_no_network"}
        result.update(status="passed", cleanup="passed")
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error_code=getattr(exc, "code", None))
        raise
    finally:
        if quota_client is not None:
            keys = list(quota_client.scan_iter(match=settings.provider_quota_namespace + ":*"))
            if keys:
                quota_client.delete(*keys)
            result["remaining_test_quota_keys"] = len(list(quota_client.scan_iter(
                match=settings.provider_quota_namespace + ":*")))
        result["elapsed_seconds"] = round(time.monotonic()-started, 3)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in {"selection", "local_inference"}},
                         ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upgrades", action="store_true", help="Include real MCP logs and Redis, mock provider HTTP")
    args = parser.parse_args()
    run(args.output, args.upgrades)
