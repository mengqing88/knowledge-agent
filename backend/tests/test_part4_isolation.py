import subprocess
from types import SimpleNamespace

import pytest
from app.repair import service as code_repair


@pytest.mark.parametrize("stdout,stderr", [("partial-output", None), (b"partial-output", b"partial-error")])
def test_container_timeout_preserves_partial_output_and_attempts_cleanup(settings, monkeypatch, stdout, stderr):
    task = SimpleNamespace(id="fixture-timeout", artifact_sha256="fixture-hash")
    monkeypatch.setattr(code_repair, "_workspace_for_task", lambda *_: None)
    monkeypatch.setattr(code_repair, "_docker_engine_identity", lambda: {"server_version": "fixture"})
    monkeypatch.setattr(code_repair, "_docker_image_id", lambda *_: "sha256:fixture")
    monkeypatch.setattr(code_repair, "_test_files", lambda *_: ["app.py"])
    monkeypatch.setattr(code_repair, "artifact_hash", lambda *_: "fixture-hash")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(code_repair.subprocess, "run", run)
    code, output, environment = code_repair.run_tests(task, settings)
    assert code == 124 and "partial-output" in output
    assert commands[-1] == ["docker", "rm", "-f", "paperqa-code-test-fixture-timeout"]
    assert environment["network"] == "none" and environment["workspace_read_only"] is True
