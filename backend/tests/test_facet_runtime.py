import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from app.core.config import Settings
from app.core.errors import BudgetExhausted, DomainError, InterruptedRun, ProviderError
from app.core.models import Chunk, IndexRevision, QuestionRun, RunEvent, Snapshot
from app.core.schemas import RunCreate
from app.jobs.services import claim_job, create_run
from app.jobs.worker import process_job
from app.knowledge import facet_retrieval, local_reranker
from app.knowledge.facet_retrieval import context_search
from app.knowledge.ingestion import index_config
from sqlalchemy import select


@pytest.fixture
def context_config(settings, monkeypatch):
    settings.retrieval_strategy = "facet_context"
    config = {name: getattr(settings, name) for name in type(settings).model_fields}
    monkeypatch.setattr(facet_retrieval, "validate_local_model", lambda config: {})
    return config


def search(db, fake, config, **options):
    return context_search(db, db.get(Snapshot, "snap1"), "M F1 dataset X", fake, "test-embedding",
        index_config(Settings(_env_file=None, embedding_model="test-embedding")), None, config,
        options.pop("cache", {}), guard=options.pop("guard", lambda: None),
        deadline=time.monotonic()+60, **options)


def test_settings_enable_new_strategy_but_test_fixture_pins_legacy(settings):
    assert Settings(_env_file=None).retrieval_strategy == "facet_context"
    assert settings.retrieval_strategy == "legacy"


def test_live_selection_keeps_sources_and_rejects_capacity_before_embedding(sessions, corpus, fake, context_config,
                                                                          monkeypatch):
    observed = []

    def rerank(requests, config, **options):
        observed.extend(requests)
        return [[c["id"] for c in request["candidates"]] for request in requests]

    monkeypatch.setattr(facet_retrieval, "rerank_local", rerank)
    with sessions() as db:
        hits = search(db, fake, context_config)
        assert [h["chunk_id"] for h in hits] == ["chunk1"]
        assert observed[0]["candidates"][0]["text"] == db.get(Chunk, "chunk1").text
        context_config["lexical_max_chars"] = 1
        monkeypatch.setattr(fake, "embed", lambda *args: pytest.fail("Capacity checked too late"))
        with pytest.raises(ProviderError) as error:
            search(db, fake, context_config)
        assert error.value.code == "lexical_capacity"


def test_cached_context_rechecks_snapshot_and_does_not_accept_foreign_sources(sessions, corpus, fake, context_config,
                                                                            monkeypatch):
    monkeypatch.setattr(facet_retrieval, "rerank_local", lambda requests, *args, **kwargs: [["chunk1"]])
    cache = {}
    with sessions() as db:
        search(db, fake, context_config, cache=cache)
        assert len(cache) == 1
    with sessions.begin() as db:
        db.get(IndexRevision, "index1").status = "failed"
    with sessions() as db, pytest.raises(DomainError, match="不可用"):
        search(db, fake, context_config, cache=cache)
    with sessions.begin() as db:
        db.get(IndexRevision, "index1").status = "ready"
        db.get(Chunk, "chunk1").source_ranges = [{"block_id": "foreign", "char_start": 0, "char_end": 1}]
    with sessions() as db, pytest.raises(DomainError, match="来源"):
        search(db, fake, context_config)


def test_workflow_selects_new_path_from_frozen_config_without_llm_reranking(sessions, settings, corpus, fake,
                                                                          context_config, monkeypatch):
    calls = []

    def rank(requests, config, **kwargs):
        calls.append(requests)
        kwargs["observe"]("local_rerank_started", {"candidate_count": 1})
        return [["chunk1"] for r in requests]

    monkeypatch.setattr(facet_retrieval, "rerank_local", rank)
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="M F1 dataset X"),
                         "context-frozen", "local-user", settings)
        run_id = run.id
    settings.retrieval_strategy = "legacy"
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        assert run.config["retrieval_strategy"] == "facet_context"
        assert run.state["sources"][0]["quote"].startswith("On dataset X")
        assert db.scalar(select(RunEvent).where(RunEvent.run_id == run_id,
                                               RunEvent.kind == "context_selection"))
    assert calls and "Reranker" not in fake.roles


def test_missing_model_fails_before_embedding(sessions, corpus, settings, fake, monkeypatch):
    config = {name: getattr(settings, name) for name in type(settings).model_fields}
    config["local_reranker_python"] = "missing-python.exe"
    monkeypatch.setattr(fake, "embed", lambda *args: pytest.fail("Must reject model first"))
    with sessions() as db, pytest.raises(ProviderError) as error:
        search(db, fake, config)
    assert error.value.code == "local_reranker_config"


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_local_process_is_killed_when_cancelled_or_timed_out(settings, monkeypatch, tmp_path, stop):
    config = {name: getattr(settings, name) for name in type(settings).model_fields}
    config["local_reranker_timeout_seconds"] = 60
    monkeypatch.setattr(local_reranker, "ROOT", tmp_path)
    monkeypatch.setattr(local_reranker, "validate_local_model", lambda c: {})
    process, count = None, 0

    class Process:
        def __init__(self, *args, **kwargs):
            nonlocal process
            process = self
            self.killed = False

        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("test", timeout)

        def poll(self):
            return 1 if self.killed else None

        def kill(self):
            self.killed = True

    def guard():
        nonlocal count
        count += 1
        if count == 4:
            raise InterruptedRun() if stop == "cancel" else BudgetExhausted("time")

    monkeypatch.setattr(local_reranker.subprocess, "Popen", Process)
    monkeypatch.setattr(local_reranker, "_stop_process", lambda process: process.kill())
    with pytest.raises(InterruptedRun if stop == "cancel" else BudgetExhausted):
        local_reranker.rerank_local([{"id": "q", "question": "test", "candidates": [{"id": "a", "text": "t"}]}],
            config, guard=guard, deadline=time.monotonic()+60)
    assert process.killed
    assert not list((tmp_path / "tmp/runtime-rerank").iterdir())


def test_local_worker_output_is_bound_to_request_and_model(settings, monkeypatch, tmp_path):
    config = {name: getattr(settings, name) for name in type(settings).model_fields}
    monkeypatch.setattr(local_reranker, "ROOT", tmp_path)
    identity = {"manifest_sha256": "frozen", "worker_sha256": "worker"}
    monkeypatch.setattr(local_reranker, "validate_local_model", lambda c: identity)
    monkeypatch.setattr(local_reranker, "local_model_identity", lambda c: identity)

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
            row = {"id": request["id"], "request_sha256": "wrong", "model_manifest_sha256": "frozen",
                   "scores": [{"id": "a", "score": 1.}]}
            Path(command[5]).write_text(json.dumps(row), encoding="utf-8")

        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(local_reranker.subprocess, "Popen", Process)
    with pytest.raises(ProviderError) as error:
        local_reranker.rerank_local([{"id": "q", "question": "test", "candidates": [{"id": "a", "text": "t"}]}],
            config, guard=lambda: None, deadline=time.monotonic()+60)
    assert error.value.code == "local_reranker_result"


@pytest.mark.skipif(os.name != "nt", reason="Windows venv launcher owns a child interpreter")
def test_windows_cancellation_terminates_the_owned_process_tree(monkeypatch):
    commands = []
    monkeypatch.setattr(local_reranker.subprocess, "run", lambda command, **kwargs: commands.append(command))
    class OwnedProcess:
        pid = 123456
        def poll(self):
            return 1
    local_reranker._stop_process(OwnedProcess())
    assert commands == [["taskkill", "/PID", "123456", "/T", "/F"]]
