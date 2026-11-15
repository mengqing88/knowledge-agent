import json

import httpx
import pytest

from scripts.demo import load_teaching_demo


@pytest.fixture
def teaching_root(tmp_path, monkeypatch):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "manifest.json").write_text(json.dumps({"fictional": True, "papers": [
        {"filename": "example.pdf", "title": "Fictional paper"}]}), encoding="utf-8")
    (fixtures / "example.pdf").write_bytes(b"fictional fixture bytes")
    monkeypatch.setattr(load_teaching_demo, "ROOT", tmp_path)
    monkeypatch.setattr(load_teaching_demo.time, "sleep", lambda _seconds: None)
    monkeypatch.delenv("KB_EVALUATION_TOKEN", raising=False)
    return tmp_path


@pytest.mark.parametrize("authentication", ["login", "token"])
def test_teaching_import_authenticates_upload_and_snapshot(teaching_root, monkeypatch, capsys, authentication):
    requests, states = [], iter(["indexing", "ready"])

    def handle(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/auth/login":
            assert json.loads(request.content) == {"username": "demo", "password": "secret-password"}
            return httpx.Response(200, json={"id": "demo-user"}, headers={"set-cookie": "kb_session=local-session; Path=/"})
        if authentication == "token":
            assert request.headers["authorization"] == "Bearer local-token"
        else:
            assert request.headers["cookie"] == "kb_session=local-session"
        if request.url.path == "/api/corpora":
            return httpx.Response(200 if request.method == "GET" else 201,
                                  json=[] if request.method == "GET" else {"id": "demo-corpus"})
        if request.url.path == "/api/corpora/demo-corpus/papers":
            assert b"fictional fixture bytes" in request.content
            return httpx.Response(202, json={"id": "demo-version"})
        if request.url.path == "/api/paper-versions/demo-version":
            return httpx.Response(200, json={"status": next(states)})
        if request.url.path == "/api/corpora/demo-corpus/snapshots":
            assert json.loads(request.content) == {"paper_version_ids": ["demo-version"]}
            return httpx.Response(201, json={"id": "demo-snapshot"})
        pytest.fail(f"Unexpected request: {request.url.path}")

    client_class = httpx.Client

    def make_client(**kwargs):
        assert kwargs["trust_env"] is False
        return client_class(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(load_teaching_demo.httpx, "Client", make_client)
    if authentication == "token":
        monkeypatch.setenv("DEMO_TEST_TOKEN", "local-token")
        monkeypatch.setattr(load_teaching_demo.getpass, "getpass", lambda *_: pytest.fail("Token must not prompt"))
        args = ["--token-env", "DEMO_TEST_TOKEN"]
    else:
        monkeypatch.setattr(load_teaching_demo.getpass, "getpass", lambda *_: "secret-password")
        args = ["--username", "demo"]
    load_teaching_demo.main(["--api", "http://testserver", *args])
    result = json.loads(capsys.readouterr().out)
    assert result["snapshot_id"] == "demo-snapshot" and result["fictional"] is True
    rows = [json.loads(line) for line in (teaching_root / "data/teaching_dev_questions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert all(row["snapshot_id"] == "demo-snapshot" and row["fictional"] and not row["human_reviewed"] for row in rows)
    assert (("POST", "/api/auth/login") in requests) == (authentication == "login")
    assert not any(secret in json.dumps(result) for secret in ("secret-password", "local-token", "local-session"))


@pytest.mark.parametrize("authentication", ["login", "token"])
def test_teaching_import_stops_on_authentication_failure(teaching_root, monkeypatch, authentication):
    requests = []

    def handle(request):
        requests.append(request.url.path)
        return httpx.Response(401, json={"detail": "Please log in"})

    client_class = httpx.Client
    monkeypatch.setattr(load_teaching_demo.httpx, "Client", lambda **kwargs: client_class(
        **kwargs, transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(load_teaching_demo.getpass, "getpass", lambda *_: "invalid-password")
    if authentication == "token":
        monkeypatch.setenv("DEMO_TEST_TOKEN", "expired-token")
        args = ["--token-env", "DEMO_TEST_TOKEN"]
    else:
        args = ["--username", "demo"]
    with pytest.raises(httpx.HTTPStatusError) as error:
        load_teaching_demo.main(["--api", "http://testserver", *args])
    assert error.value.response.status_code == 401
    assert requests == (["/api/auth/login"] if authentication == "login" else ["/api/corpora"])
    assert not (teaching_root / "data").exists()
