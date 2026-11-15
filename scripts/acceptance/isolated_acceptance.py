"""Disposable integration check; requires deploy/compose.acceptance.yaml already running.

Uses real PostgreSQL, Redis, Milvus and independent Celery/beat processes. The
model provider is the deterministic test double, never an external endpoint.
The caller owns the uniquely named Compose project and its eventual cleanup.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "backend" / "tests")]


def port(project, service, target):
    value = subprocess.check_output(["docker", "compose", "-p", project, "-f",
        str(ROOT / "deploy" / "compose.acceptance.yaml"), "port", service, str(target)], text=True).strip()
    return int(value.rsplit(":", 1)[1])


def wait_for(get_value, desired, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = get_value()
        if value.get("status") == desired:
            return value
        if value.get("status") in ("failed", "cancelled", "rejected"):
            raise AssertionError(f"unexpected terminal state: {value.get('status')}, {value.get('error')}")
        time.sleep(.5)
    raise TimeoutError(f"waiting for {desired}: {value.get('status')}, {value.get('error')}")


def celery_process(kind, schedule):
    if os.environ.get("KB_ACCEPTANCE_CHILD") != "1":
        raise SystemExit("internal child mode requires the isolated acceptance parent")
    from app.integrations import providers
    from app.jobs import tasks
    from conftest import FakeProvider
    class AcceptanceProvider(FakeProvider):
        def structured(self, role, prompt, data, schema):
            marker = Path(os.environ["ACCEPTANCE_PAUSE_MARKER"])
            if role == "Planner" and data.get("question") == "验收恢复：M 的 F1 是多少？" and not marker.exists():
                marker.write_text("test worker reached Planner", encoding="utf-8")
                # The acceptance parent deliberately terminates this worker.
                while True:
                    time.sleep(.1)
            time.sleep(.3)
            return super().structured(role, prompt, data, schema)

    providers.ModelProvider = lambda _settings: AcceptanceProvider()
    if kind == "worker":
        tasks.celery_app.worker_main(["worker", "--pool=solo", "--loglevel=WARNING"])
    else:
        tasks.celery_app.start(["beat", "--loglevel=WARNING", "--schedule", schedule])


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def acceptance_environment(project, temp_path):
    env = os.environ.copy()
    env.update({
            "DATABASE_URL": f"postgresql+psycopg://acceptance:acceptance-local-only@127.0.0.1:{port(project, 'db', 5432)}/acceptance",
            "CELERY_BROKER_URL": f"redis://127.0.0.1:{port(project, 'redis', 6379)}/0",
            "MILVUS_URI": f"http://127.0.0.1:{port(project, 'milvus', 19530)}",
            "STORAGE_DIR": str(temp_path / "documents"), "TASK_BACKEND": "celery",
            "CHAT_API_KEY": "local-fake", "CHAT_BASE_URL": "http://127.0.0.1:1", "MODEL": "fake",
            "EMBEDDING_API_KEY": "local-fake", "EMBEDDING_BASE_URL": "http://127.0.0.1:1",
            "EMBEDDING_MODEL": "test-embedding", "EMBEDDING_DIMENSIONS": "3",
            "ROLE_MODEL_ROUTES": "{}", "RERANKER_BACKEND": "llm", "QUERY_REWRITE_ENABLED": "false",
            "WORKFLOW_MCP_ENABLED": "false", "PROJECT_LOGS_MCP_ENABLED": "false",
            "MCP_ENABLED": "false", "MCP_WRITES_ENABLED": "false",
            "PROVIDER_QUOTA_ENABLED": "false", "ANSWER_REVIEW_REQUIRED": "true",
            "MILVUS_COLLECTION_PREFIX": "kbaccept", "MILVUS_DB_NAME": "default", "MILVUS_TOKEN": "",
            "ACCEPTANCE_PAUSE_MARKER": str(temp_path / "worker-paused"),
            "KB_ACCEPTANCE_CHILD": "1", "CODE_REPOSITORY_ROOT": str(temp_path / "repositories"),
            "CODE_TASK_WORKSPACE_DIR": str(temp_path / "code-tasks"),
    })
    return env


def run_versions(project):
    with tempfile.TemporaryDirectory(prefix="kb-versions-acceptance-") as temp:
        os.environ.update(acceptance_environment(project, Path(temp)))
        from app.core.config import get_settings
        from app.core.db import Session

        from scripts.acceptance.acceptance_concurrency import check_version_switch

        check_version_switch(Session, get_settings())
        return {"status": "passed", "project": project, "model": "deterministic FakeProvider; no model HTTP",
                "checks": ["two-concurrent-PG-Milvus-indexing-jobs", "failed-update-preserves-active-version"]}


def run(project, browser_port=None, browser_stop=None):
    with tempfile.TemporaryDirectory(prefix="kb-acceptance-") as temp:
        temp_path = Path(temp)
        env = acceptance_environment(project, temp_path)
        os.environ.update(env)
        subprocess.run([sys.executable, "-X", "utf8", "scripts/manage.py", "migrate"], cwd=ROOT,
                       env=env, check=True)
        from app.api.main import app
        from app.core.auth import hash_password
        from app.core.db import Session
        from app.core.models import AnswerRevision, ModelCall, QuestionRun, RunEvent, User, now
        from fastapi.testclient import TestClient
        from sqlalchemy import func, select, text

        prefix = f"accept-{uuid4().hex[:8]}"
        users = {role: f"{prefix}-{role}" for role in ("admin", "reader", "editor")}
        with Session.begin() as db:
            db.add(User(id=users["admin"], username=users["admin"], password_hash=hash_password("accept-pass")))
        logs = open(temp_path / "celery.log", "w", encoding="utf-8")
        processes = []
        try:
            def start_child(kind):
                process = subprocess.Popen([sys.executable, "-X", "utf8", __file__, "--child", kind,
                    "--schedule", str(temp_path / "beat-schedule")], cwd=ROOT, env=env, stdout=logs,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                processes.append(process)
                return process

            worker = start_child("worker")
            start_child("beat")
            with TestClient(app) as client:
                def post(path, **kwargs):
                    response = client.post(path, **kwargs)
                    assert response.is_success, f"POST {path}: {response.status_code} {response.text}"
                    return response.json()

                def login(user):
                    return post("/api/auth/login", json={"username": user, "password": "accept-pass"})

                def get(path):
                    response = client.get(path)
                    assert response.is_success, f"GET {path}: {response.status_code} {response.text}"
                    return response.json()

                login(users["admin"])
                corpus = post("/api/corpora", json={"name": "Isolated acceptance"})["id"]
                member_ids = {}
                for role in ("reader", "editor"):
                    member_ids[role] = post(f"/api/corpora/{corpus}/members", json={"username": users[role],
                         "password": "accept-pass", "role": role})["id"]
                content = b"On dataset X, M achieved F1 82 and baseline F1 80.\n"
                version = post(f"/api/corpora/{corpus}/papers", files={"file": ("test.txt", content)})["id"]
                wait_for(lambda: get(f"/api/paper-versions/{version}"), "ready")
                snapshot = post(f"/api/corpora/{corpus}/snapshots", json={"paper_version_ids": [version]})["id"]
                login(users["reader"])
                request = {"snapshot_id": snapshot, "question": "M 的 F1 是多少？"}
                headers = {"Idempotency-Key": f"{prefix}-1"}
                run_id = post("/api/question-runs", json=request, headers=headers)["id"]
                assert post("/api/question-runs", json=request, headers=headers)["id"] == run_id
                draft = wait_for(lambda: get(f"/api/question-runs/{run_id}"), "review_pending")
                assert draft["answer"] is None and draft["answer_sha256"] is None
                progress = get(f"/api/question-runs/{run_id}/events?stream=false")
                assert progress and "数据集 X 上 M 的 F1 为 82" not in str(progress)
                login(users["editor"])
                view = get(f"/api/question-runs/{run_id}")
                answer = view["answer"]
                assert answer["evidence_status"] == "supported" and answer["claims"]
                source = answer["evidence"][0]
                assert source["paper_version_id"] == version and source["sha256"] == hashlib.sha256(content).hexdigest()
                assert client.get(f"/api/question-runs/{run_id}/events?stream=false").status_code == 404
                post(f"/api/answers/{answer['id']}/approval", json={"decision": "accept",
                     "answer_sha256": view["answer_sha256"]})
                login(users["reader"])
                assert get(f"/api/question-runs/{run_id}")["status"] == "completed"
                assert client.get(source["file_url"]).content == content
                resumed = get(f"/api/question-runs/{run_id}/events?stream=false&after_seq={progress[-1]['seq']}")
                assert [event["kind"] for event in resumed] == ["completed"]
                with Session() as db:
                    assert db.scalar(select(func.count()).select_from(AnswerRevision).where(
                        AnswerRevision.run_id == run_id)) == 1
                    checkpoints = db.execute(text("SELECT count(*) FROM checkpoints WHERE thread_id = :thread"),
                                             {"thread": f"{run_id}:1"}).scalar_one()
                    assert checkpoints > 0
                print("PASS: upload, isolated PG/Milvus indexing, Redis/Celery/beat, PostgresSaver, "
                      "idempotency, review visibility/hash, citation and event replay", flush=True)

                recovery_id = post("/api/question-runs", json={**request,
                    "question": "验收恢复：M 的 F1 是多少？"}, headers={"Idempotency-Key": f"{prefix}-recovery"})["id"]
                deadline = time.monotonic() + 60
                marker = Path(env["ACCEPTANCE_PAUSE_MARKER"])
                while not marker.exists() and time.monotonic() < deadline:
                    assert worker.poll() is None, "test worker exited before interruption"
                    time.sleep(.1)
                assert marker.exists(), "worker did not reach the interruption point"
                stop_processes([worker])
                from datetime import timedelta

                with Session.begin() as db:
                    interrupted = db.get(QuestionRun, recovery_id)
                    assert interrupted.status == "running" and interrupted.attempt == 1
                    # Advance the lease for a bounded test; do not wait for the production timeout.
                    interrupted.lease_until = now() - timedelta(seconds=1)
                worker = start_child("worker")
                recovered = wait_for(lambda: get(f"/api/question-runs/{recovery_id}"), "review_pending")
                assert recovered["answer"] is None
                with Session() as db:
                    assert db.get(QuestionRun, recovery_id).attempt == 2
                    assert db.scalar(select(func.count()).select_from(AnswerRevision).where(
                        AnswerRevision.run_id == recovery_id)) == 1
                    attempts = db.scalars(select(RunEvent.attempt).where(RunEvent.run_id == recovery_id,
                        RunEvent.kind == "started").order_by(RunEvent.seq)).all()
                    assert attempts == [1, 2], attempts
                    assert db.scalars(select(ModelCall.attempt).where(ModelCall.run_id == recovery_id)).all()
                    assert db.scalar(text("SELECT count(*) FROM checkpoints WHERE thread_id = :thread"),
                        {"thread": f"{recovery_id}:2"}) > 0
                print("PASS: killed independent Celery worker, forced lease expiry, restarted worker, one answer", flush=True)

                stop_processes(processes)
                from app.core.config import get_settings

                from scripts.acceptance.acceptance_concurrency import check_concurrency

                check_concurrency(Session, get_settings(), snapshot, corpus, member_ids["reader"],
                                  member_ids["editor"], version)
                if browser_port:
                    start_child("worker")
                    start_child("beat")
                    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1",
                        "--port", str(browser_port)], cwd=ROOT / "backend", env=env, stdout=logs,
                        stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    processes.append(api)
                    print(json.dumps({"browser_url": f"http://127.0.0.1:{browser_port}", "users": users,
                        "password": "accept-pass", "corpus": corpus}, ensure_ascii=False), flush=True)
                    deadline = time.monotonic() + 900
                    while time.monotonic() < deadline and not Path(browser_stop).exists():
                        assert api.poll() is None, "acceptance API exited"
                        time.sleep(.5)
                return {"status": "passed", "project": project, "model": "deterministic FakeProvider; no model HTTP",
                        "checks": ["upload-index-ask-review-citation-replay", "celery-process-interruption-recovery",
                                   "eight-concurrent-claimants-and-stale-fence", "approval-vs-revocation-and-deactivation"],
                        "limits": ["forced lease expiry", "synthetic TXT", "business quality not evaluated"]}
        except Exception:
            logs.flush()
            print("Celery test log (last 4000 characters):", file=sys.stderr)
            print((temp_path / "celery.log").read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
            raise
        finally:
            stop_processes(processes)
            logs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    parser.add_argument("--child", choices=("worker", "beat"))
    parser.add_argument("--schedule")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--browser-port", type=int)
    parser.add_argument("--browser-stop", type=Path)
    parser.add_argument("--versions-only", action="store_true", help="Run after the flow check; workers must be stopped")
    args = parser.parse_args()
    if args.child:
        celery_process(args.child, args.schedule)
    elif args.project and args.project.startswith("kbaccept-"):
        if args.browser_port and not args.browser_stop:
            parser.error("--browser-port requires --browser-stop")
        report = (run_versions(args.project) if args.versions_only else
                  run(args.project, args.browser_port, args.browser_stop))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        parser.error("provide an isolated Compose project beginning with kbaccept-")
