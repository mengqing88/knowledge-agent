import subprocess
import sys

import pytest
from app import code_api, code_repair
from app import main as main_module
from app.code_repair import _validate_patch
from app.db import get_db
from app.errors import DomainError
from app.main import app
from app.models import CodeRepairTask, RepairExperience
from fastapi.testclient import TestClient
from sqlalchemy import select


@pytest.fixture
def client(sessions, settings, corpus, tmp_path, monkeypatch):
    def override():
        with sessions() as db:
            yield db

    repo_root = tmp_path / "repositories"
    repo = repo_root / "sample"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text(
        "import os\nimport socket\nfrom pathlib import Path\n\nimport pytest\n\n"
        "from app import value\n\n\ndef test_value():\n    assert value == 2\n\n\n"
        "def test_container_is_read_only_and_has_no_secrets_or_network():\n"
        "    assert not Path('/workspace/.git').exists()\n"
        "    assert not Path('/workspace/.env').exists()\n"
        "    assert 'CHAT_API_KEY' not in os.environ\n"
        "    with pytest.raises(OSError):\n"
        "        socket.create_connection(('1.1.1.1', 443), timeout=0.5)\n"
        "    with pytest.raises(OSError):\n"
        "        Path('/workspace/.write-probe').write_text('blocked')\n",
        encoding="utf-8",
    )
    (repo / ".env").write_text("CHAT_API_KEY=synthetic-secret\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Repair Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)

    settings.code_repository_root = repo_root
    settings.code_task_workspace_dir = tmp_path / "task-workspaces"
    monkeypatch.setattr(code_api, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "Session", sessions)
    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as test_client:
            assert test_client.post("/api/auth/login", json={
                "username": "admin", "password": "test-password"}).status_code == 200
            yield test_client, repo
    finally:
        app.dependency_overrides.clear()


def patch_value(old="1", new="2"):
    return (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        f"-value = {old}\n"
        f"+value = {new}\n"
    )


class FakeRepairProvider:
    def __init__(self, patch):
        self.patch = patch

    def structured(self, role, prompt, data, schema):
        assert role == "CodeRepair"
        assert data["selected_files"][0]["path"] == "app.py"
        return schema.model_validate({"summary": "Fix the expected value.", "diff": self.patch}), {}


def create_task(client, repo):
    response = client.post("/api/code-repair/tasks", json={
        "repository_path": "sample",
        "base_ref": "HEAD",
        "objective": "Set the value required by the regression test.",
    })
    assert response.status_code == 201, response.text
    task = response.json()
    workspace = task["id"]
    task_dir = repo.parent.parent / "task-workspaces" / "local-user" / workspace
    return task, task_dir


def test_task_uses_clean_commit_pinned_detached_worktree(client):
    client, repo = client
    task, worktree = create_task(client, repo)
    assert task["status"] == "ready"
    assert (worktree / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    head = subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip()
    assert head == task["base_commit"]
    symbolic = subprocess.run(["git", "-C", str(worktree), "symbolic-ref", "-q", "HEAD"])
    assert symbolic.returncode != 0


def test_restore_checkpoint_recovers_modified_deleted_and_untracked_files(
        client, sessions, settings):
    test_client, repo = client
    task, worktree = create_task(test_client, repo)
    (worktree / "app.py").write_text("value = 9\n", encoding="utf-8")
    (worktree / "tests" / "test_app.py").unlink()
    (worktree / "untracked.txt").write_text("temporary", encoding="utf-8")

    with sessions() as db:
        record = db.get(CodeRepairTask, task["id"])
        code_repair.restore_checkpoint(record, settings)

    assert (worktree / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (worktree / "tests" / "test_app.py").is_file()
    assert not (worktree / "untracked.txt").exists()
    assert subprocess.check_output(
        ["git", "-C", str(worktree), "status", "--porcelain"], text=True,
    ).strip() == ""
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_dirty_source_repository_is_rejected(client):
    client, repo = client
    (repo / "untracked.txt").write_text("preserve me", encoding="utf-8")
    response = client.post("/api/code-repair/tasks", json={
        "repository_path": "sample", "base_ref": "HEAD", "objective": "Fix a regression.",
    })
    assert response.status_code == 409
    assert response.json()["code"] == "repository_dirty"
    assert (repo / "untracked.txt").read_text(encoding="utf-8") == "preserve me"


@pytest.mark.parametrize("path", ["tests/test_app.py", "pyproject.toml", ".github/workflows/checks.yml"])
def test_test_and_acceptance_files_are_rejected_by_patch_guard(path):
    diff = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n"
    )
    with pytest.raises(DomainError) as exc:
        _validate_patch(diff, [path])
    assert exc.value.code in {"protected_file", "patch_scope"}


def test_repository_paths_cannot_escape_configured_root(client):
    test_client, _repo = client
    response = test_client.post("/api/code-repair/tasks", json={
        "repository_path": "../outside",
        "base_ref": "HEAD",
        "objective": "Do not inspect outside the configured root.",
    })
    assert response.status_code == 403
    assert response.json()["code"] == "repository_path"


def test_protected_test_file_cannot_be_patched(client, monkeypatch):
    test_client, repo = client
    monkeypatch.setattr(code_api, "ModelProvider", lambda _settings: FakeRepairProvider(
        "diff --git a/tests/test_app.py b/tests/test_app.py\n"
        "--- a/tests/test_app.py\n"
        "+++ b/tests/test_app.py\n"
        "@@ -1,4 +1,4 @@\n"
        "-from app import value\n"
        "+from app import value\n"
    ))
    task, worktree = create_task(test_client, repo)
    response = test_client.post(f"/api/code-repair/tasks/{task['id']}/propose", json={"files": ["app.py"]})
    assert response.status_code == 400
    assert response.json()["code"] == "patch_scope"
    assert (worktree / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_passed_artifact_can_be_accepted_and_experience_is_version_bound(client, sessions, monkeypatch):
    test_client, repo = client
    monkeypatch.setattr(code_api, "ModelProvider", lambda _settings: FakeRepairProvider(patch_value()))
    task, worktree = create_task(test_client, repo)
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_app.py::test_value"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert baseline.returncode != 0
    assert "AssertionError" in baseline.stdout + baseline.stderr
    files = test_client.get(f"/api/code-repair/tasks/{task['id']}/files")
    assert files.json()["files"] == ["app.py"]
    proposed = test_client.post(f"/api/code-repair/tasks/{task['id']}/propose",
                                json={"files": ["app.py"]})
    assert proposed.status_code == 200, proposed.text
    assert proposed.json()["status"] == "patched"
    tested = test_client.post(f"/api/code-repair/tasks/{task['id']}/test")
    assert tested.status_code == 200, tested.text
    assert tested.json()["status"] == "review_pending"
    assert tested.json()["test_exit_code"] == 0
    accepted = test_client.post(f"/api/code-repair/tasks/{task['id']}/review", json={
        "decision": "accept",
    })
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["artifact_sha256"] == proposed.json()["patch_sha256"]
    with sessions() as db:
        experience = db.scalar(select(RepairExperience).where(RepairExperience.task_id == task["id"]))
        assert experience.artifact_sha256 == proposed.json()["patch_sha256"]
        assert experience.verified_command == "python -m pytest -q"
    assert tested.json()["test_environment"]["network"] == "none"
    assert tested.json()["test_environment"]["workspace_read_only"] is True
    assert tested.json()["test_environment"]["image_id"].startswith("sha256:")
    assert tested.json()["test_environment"]["docker_engine"]["server_version"]
    assert accepted.json()["bundle_sha256"] == tested.json()["verification_sha256"]
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (worktree / "app.py").read_text(encoding="utf-8") == "value = 2\n"


def test_rejection_requires_reason_and_restores_worktree(client, sessions, monkeypatch):
    test_client, repo = client
    monkeypatch.setattr(code_api, "ModelProvider", lambda _settings: FakeRepairProvider(patch_value()))
    task, worktree = create_task(test_client, repo)
    test_client.post(f"/api/code-repair/tasks/{task['id']}/propose", json={"files": ["app.py"]})
    assert test_client.post(f"/api/code-repair/tasks/{task['id']}/test").json()["status"] == "review_pending"
    missing_reason = test_client.post(f"/api/code-repair/tasks/{task['id']}/review", json={
        "decision": "reject",
    })
    assert missing_reason.status_code == 422
    rejected = test_client.post(f"/api/code-repair/tasks/{task['id']}/review", json={
        "decision": "reject", "reason": "The change is too broad.",
    })
    assert rejected.status_code == 200
    assert rejected.json()["decision"] == "reject"
    assert (worktree / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    with sessions() as db:
        assert db.scalar(select(RepairExperience).where(RepairExperience.task_id == task["id"])) is None
        assert db.get(CodeRepairTask, task["id"]).status == "rejected"


def test_rejection_feedback_is_durable_when_workspace_cleanup_fails(client, sessions, monkeypatch):
    test_client, repo = client
    monkeypatch.setattr(code_api, "ModelProvider", lambda _settings: FakeRepairProvider(patch_value()))
    task, worktree = create_task(test_client, repo)
    test_client.post(f"/api/code-repair/tasks/{task['id']}/propose", json={"files": ["app.py"]})
    assert test_client.post(f"/api/code-repair/tasks/{task['id']}/test").json()["status"] == "review_pending"

    original_restore = code_repair.restore_checkpoint
    monkeypatch.setattr(code_api, "restore_checkpoint", lambda *_args: (_ for _ in ()).throw(OSError("offline")))
    review = {"decision": "reject", "reason": "The reported fix is too broad.",
              "feedback_category": "missing_scenario", "root_cause_layer": "workflow",
              "uncertainty": "Need to confirm which caller is affected."}
    failed_cleanup = test_client.post(f"/api/code-repair/tasks/{task['id']}/review", json=review)
    assert failed_cleanup.status_code == 503
    with sessions() as db:
        stored = db.scalar(select(code_api.CodeRepairReview).where(
            code_api.CodeRepairReview.task_id == task["id"]))
        assert stored.reason == review["reason"]
        assert stored.feedback_category == "missing_scenario"
        assert stored.uncertainty == review["uncertainty"]
        assert db.get(CodeRepairTask, task["id"]).workspace_state == "cleanup_failed"
    fetched_feedback = test_client.get(f"/api/code-repair/tasks/{task['id']}/review")
    assert fetched_feedback.json()["reason"] == review["reason"]
    assert fetched_feedback.json()["root_cause_layer"] == "workflow"

    monkeypatch.setattr(code_api, "restore_checkpoint", original_restore)
    cleaned = test_client.post(f"/api/code-repair/tasks/{task['id']}/cleanup")
    assert cleaned.status_code == 200
    assert cleaned.json()["workspace_state"] == "restored"
    assert (worktree / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_changed_test_evidence_cannot_be_approved(client, sessions, monkeypatch):
    test_client, repo = client
    monkeypatch.setattr(code_api, "ModelProvider", lambda _settings: FakeRepairProvider(patch_value()))
    task, _worktree = create_task(test_client, repo)
    test_client.post(f"/api/code-repair/tasks/{task['id']}/propose", json={"files": ["app.py"]})
    assert test_client.post(f"/api/code-repair/tasks/{task['id']}/test").json()["status"] == "review_pending"
    with sessions() as db:
        record = db.get(CodeRepairTask, task["id"])
        record.test_output += "changed after verification"
        db.commit()
    response = test_client.post(f"/api/code-repair/tasks/{task['id']}/review", json={"decision": "accept"})
    assert response.status_code == 409
    assert response.json()["code"] == "verification_changed"


def test_failed_tests_restore_files_and_preserve_failure_evidence(client, monkeypatch):
    test_client, repo = client
    failing_patch = patch_value(new="3")
    monkeypatch.setattr(code_api, "ModelProvider", lambda _settings: FakeRepairProvider(failing_patch))
    task, worktree = create_task(test_client, repo)
    test_client.post(f"/api/code-repair/tasks/{task['id']}/propose", json={"files": ["app.py"]})
    tested = test_client.post(f"/api/code-repair/tasks/{task['id']}/test")
    assert tested.json()["status"] == "failed"
    assert tested.json()["test_exit_code"] != 0
    assert "assert" in tested.json()["test_output"]
    assert (worktree / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_changed_artifact_cannot_be_approved(client, monkeypatch):
    test_client, repo = client
    monkeypatch.setattr(code_api, "ModelProvider", lambda _settings: FakeRepairProvider(patch_value()))
    task, worktree = create_task(test_client, repo)
    test_client.post(f"/api/code-repair/tasks/{task['id']}/propose", json={"files": ["app.py"]})
    test_client.post(f"/api/code-repair/tasks/{task['id']}/test")
    (worktree / "app.py").write_text("value = 99\n", encoding="utf-8")
    response = test_client.post(f"/api/code-repair/tasks/{task['id']}/review", json={
        "decision": "accept",
    })
    assert response.status_code == 409
    refreshed = test_client.get(f"/api/code-repair/tasks/{task['id']}")
    assert refreshed.json()["status"] == "failed"
    assert (worktree / "app.py").read_text(encoding="utf-8") == "value = 1\n"
