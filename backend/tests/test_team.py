import hashlib
from copy import deepcopy
from datetime import timedelta
from io import BytesIO

import pytest
from app import main as main_module
from app import services as services_module
from app.access import require_role
from app.auth import hash_password, verify_password
from app.db import get_db
from app.errors import DomainError
from app.governance import digest
from app.ingestion import build_chunks, extract_document, ingest_version
from app.main import app
from app.models import (
    AcceptedAnswerExperience,
    AnswerApprovalRecord,
    AnswerClaim,
    AnswerRevision,
    Chunk,
    ClaimEvidence,
    Corpus,
    EvidenceSpan,
    LoginSession,
    Paper,
    PaperVersion,
    QuestionRun,
    SourceBlock,
    User,
    now,
)
from app.schemas import Reranking, RunCreate
from app.services import answer_content_hash, claim_job, create_run, create_snapshot
from app.workflow import Workflow, initial_state
from docx import Document
from fastapi.testclient import TestClient
from sqlalchemy import select


@pytest.fixture
def client(sessions, settings, corpus, monkeypatch):
    def override():
        with sessions() as db:
            yield db
    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(main_module, "Session", sessions)
    monkeypatch.setattr(main_module, "settings", settings)
    try:
        with TestClient(app) as client:
            assert client.post("/api/auth/login", json={
                "username": "admin", "password": "test-password"}).status_code == 200
            yield client
    finally:
        app.dependency_overrides.clear()


def login_as(client, username):
    assert client.post("/api/auth/login", json={"username": username, "password": "test-password"}).status_code == 200


def test_source_file_returns_only_the_verified_version(client, sessions, settings):
    content = b"%PDF-1.4\noriginal"
    path = settings.storage_dir / "a.pdf"
    path.write_bytes(content)
    with sessions.begin() as db:
        version = db.get(PaperVersion, "v1")
        version.sha256 = hashlib.sha256(content).hexdigest()
        version.byte_size = len(content)

    response = client.get("/api/paper-versions/v1/file")
    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    etag = response.headers["etag"]
    ranged = client.get("/api/paper-versions/v1/file", headers={"Range": "bytes=0-3", "If-Range": etag})
    assert ranged.status_code == 206 and ranged.content == content[:4]
    assert ranged.headers["content-range"] == f"bytes 0-3/{len(content)}"
    assert client.get("/api/paper-versions/v1/file", headers={"Range": "bytes=-4"}).content == content[-4:]
    assert client.get("/api/paper-versions/v1/file", headers={"Range": "bytes=4-"}).content == content[4:]
    unsatisfiable = client.get("/api/paper-versions/v1/file", headers={"Range": "bytes=999-"})
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == f"bytes */{len(content)}"
    full = client.get("/api/paper-versions/v1/file", headers={"Range": "bytes=0-3", "If-Range": '"old"'})
    assert full.status_code == 200 and full.content == content
    multi = client.get("/api/paper-versions/v1/file", headers={"Range": "bytes=0-3,5-8"})
    assert multi.status_code == 200 and multi.content == content

    altered = b"%PDF-1.4\ntampered"
    assert len(altered) == len(content)
    path.write_bytes(altered)
    response = client.get("/api/paper-versions/v1/file")
    assert response.status_code == 409
    assert altered not in response.content
    assert client.get("/api/paper-versions/v1/file", headers={"Range": "bytes=0-3"}).status_code == 409

    path.unlink()
    assert client.get("/api/paper-versions/v1/file").status_code == 404


def test_non_pdf_source_is_downloaded_with_encoded_filename(client, sessions, settings):
    content = "文档内容".encode()
    (settings.storage_dir / "a.pdf").write_bytes(content)
    with sessions.begin() as db:
        version = db.get(PaperVersion, "v1")
        version.filename = "中文.txt"
        version.sha256 = hashlib.sha256(content).hexdigest()
        version.byte_size = len(content)
    response = client.get("/api/paper-versions/v1/file")
    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-disposition"] == "attachment; filename*=utf-8''%E4%B8%AD%E6%96%87.txt"


def test_password_session_expiry_and_logout(client, sessions):
    assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong-password"}).status_code == 401
    with sessions.begin() as db:
        user = db.get(User, "local-user")
        assert verify_password("test-password", user.password_hash)
        assert not verify_password("other-password", user.password_hash)
        assert hash_password("test-password") != user.password_hash
        for session in db.scalars(select(LoginSession)):
            session.expires_at = now() - timedelta(seconds=1)
    assert client.get("/api/auth/me").status_code == 401
    login_as(client, "admin")
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/corpora").status_code == 401


def test_reader_can_query_but_cannot_write_or_manage_and_revocation_applies(client, sessions, settings):
    r = client.post("/api/corpora/corpus1/members", json={
        "username": "reader", "password": "test-password", "role": "reader"})
    assert r.status_code == 201
    reader_id = r.json()["id"]
    login_as(client, "reader")
    assert client.get("/api/corpora").json()[0]["role"] == "reader"
    assert client.get("/api/corpora/corpus1/papers").status_code == 200
    assert client.get("/api/corpora/corpus1/members").status_code == 403
    assert client.post("/api/corpora/corpus1/papers", files={"file": ("x.md", b"# private")}).status_code == 403
    assert client.patch("/api/documents/paper1", json={"title": "hack"}).status_code == 403
    assert client.delete("/api/documents/paper1").status_code == 403
    snapshot = client.post("/api/corpora/corpus1/snapshots", json={"paper_version_ids": ["v1"]}).json()
    response = client.post("/api/question-runs", headers={"Idempotency-Key": "reader-run-1"}, json={
        "snapshot_id": snapshot["id"], "question": "M 的 F1 是多少？"})
    assert response.status_code == 202
    run_id = response.json()["id"]
    login_as(client, "admin")
    assert client.get(f"/api/question-runs/{run_id}").status_code == 404
    assert client.delete(f"/api/corpora/corpus1/members/{reader_id}").status_code == 200
    login_as(client, "reader")
    for path in ("/api/corpora/corpus1/papers", "/api/paper-versions/v1/file", f"/api/question-runs/{run_id}",
                 f"/api/question-runs/{run_id}/trace", f"/api/question-runs/{run_id}/events?stream=false"):
        assert client.get(path).status_code == 404
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "cancelled"
        with pytest.raises(DomainError):
            require_role(db, "corpus1", reader_id)


def test_editor_document_metadata_update_success_and_deleted_versions_not_queried(client, sessions, settings, fake):
    client.post("/api/corpora/corpus1/members", json={
        "username": "editor", "password": "test-password", "role": "editor"})
    login_as(client, "editor")
    assert client.post("/api/corpora/corpus1/members", json={"username": "x", "role": "reader"}).status_code == 403
    assert client.patch("/api/documents/paper1", json={"title": "团队手册", "folder": "入职", "tags": ["权限"]}).status_code == 200
    body = "# 权限申请\n\n在工单平台提交申请，由团队负责人审核。\n".encode()
    response = client.post("/api/corpora/corpus1/papers", data={"paper_id": "paper1"}, files={"file": ("onboarding.md", body)})
    assert response.status_code == 202
    new_id = response.json()["id"]
    assert not response.json()["is_active"]
    with sessions() as db:
        assert db.get(PaperVersion, "v1").is_active  # old content remains until successful replacement
    job = claim_job(sessions, settings)
    assert job[1] == new_id
    ingest_version(sessions, settings, fake, new_id, job[2])
    rows = client.get("/api/corpora/corpus1/papers").json()
    assert len(rows) == 1 and rows[0]["id"] == new_id
    assert rows[0]["folder"] == "入职" and rows[0]["tags"] == ["权限"]
    assert len(client.get("/api/corpora/corpus1/papers?include_history=true").json()) == 2
    assert client.post("/api/corpora/corpus1/snapshots", json={"paper_version_ids": ["v1"]}).status_code == 409
    snap = client.post("/api/corpora/corpus1/snapshots", json={"paper_version_ids": [new_id]}).json()
    assert client.delete("/api/documents/paper1").status_code == 200
    assert client.get("/api/corpora/corpus1/papers").json() == []
    assert client.post("/api/question-runs", headers={"Idempotency-Key": "deleted-run-1"}, json={
        "snapshot_id": snap["id"], "question": "权限怎么申请？"}).status_code == 409


def test_cross_knowledge_base_and_owner_role_protection(client, sessions):
    with sessions.begin() as db:
        db.add(Corpus(id="private-team", owner_id="someone-else", name="其他团队"))
    assert client.get("/api/corpora/private-team/papers").status_code == 404
    assert client.post("/api/corpora/private-team/papers", files={"file": ("x.md", b"text")}).status_code == 404
    assert client.patch("/api/corpora/corpus1/members/local-user", json={"role": "reader"}).status_code == 409
    assert client.delete("/api/corpora/corpus1/members/local-user").status_code == 409


def test_followup_same_snapshot_and_replay_keep_original_run(client, sessions):
    first = client.post("/api/question-runs", headers={"Idempotency-Key": "first-run-key"}, json={
        "snapshot_id": "snap1", "question": "M 的 F1 是多少？"}).json()
    client.post(f"/api/question-runs/{first['id']}/cancel")
    followup = client.post("/api/question-runs", headers={"Idempotency-Key": "followup-same"}, json={
        "snapshot_id": "snap1", "question": "相对基线怎样？", "parent_run_id": first["id"]})
    assert followup.status_code == 202
    other = client.post("/api/corpora/corpus1/snapshots", json={"paper_version_ids": ["v1"]}).json()
    assert client.post("/api/question-runs", headers={"Idempotency-Key": "followup-wrong"}, json={
        "snapshot_id": other["id"], "question": "相对基线怎样？", "parent_run_id": first["id"]}).status_code == 409
    replay = client.post(f"/api/question-runs/{first['id']}/replay", headers={"Idempotency-Key": "replay-test-1"})
    assert replay.status_code == 202 and replay.json()["id"] != first["id"]
    assert client.get(f"/api/question-runs/{first['id']}").json()["status"] == "cancelled"


@pytest.mark.parametrize("outcome", ["reviewed_after_recovery", "revoked_while_queued"])
def test_uploaded_document_question_lifecycle(client, sessions, settings, fake, outcome):
    settings.answer_review_required = True
    for username, role in (("lifecycle-author", "reader"), ("lifecycle-reviewer", "editor")):
        response = client.post("/api/corpora/corpus1/members", json={
            "username": username, "password": "test-password", "role": role})
        assert response.status_code == 201, response.text
        if role == "reader":
            author_id = response.json()["id"]

    # Exercise the real upload, parser, chunker and persisted ingestion path.
    # Only model inference and vector search use the deterministic test backend.
    content = b"On dataset X, M achieved F1 82 and baseline F1 80.\n"
    uploaded = client.post("/api/corpora/corpus1/papers", files={"file": ("lifecycle.txt", content)})
    assert uploaded.status_code == 202, uploaded.text
    version_id = uploaded.json()["id"]
    ingest_job = claim_job(sessions, settings)
    assert ingest_job[:2] == ("ingest", version_id)
    from app.worker import process_job
    process_job(sessions, settings, fake, ingest_job)
    ready = client.get(f"/api/paper-versions/{version_id}").json()
    assert ready["status"] == "ready" and ready["is_active"]
    snapshot = client.post("/api/corpora/corpus1/snapshots", json={"paper_version_ids": [version_id]})
    assert snapshot.status_code == 201, snapshot.text
    snapshot_id = snapshot.json()["id"]

    login_as(client, "lifecycle-author")
    body = {"snapshot_id": snapshot_id, "question": "M 的 F1 是多少？"}
    headers = {"Idempotency-Key": "uploaded-lifecycle-run"}
    queued = client.post("/api/question-runs", json=body, headers=headers)
    assert queued.status_code == 202, queued.text
    run_id = queued.json()["id"]
    assert client.post("/api/question-runs", json=body, headers=headers).json()["id"] == run_id

    if outcome == "revoked_while_queued":
        login_as(client, "admin")
        assert client.delete(f"/api/corpora/corpus1/members/{author_id}").status_code == 200
        assert claim_job(sessions, settings) is None
        with sessions() as db:
            assert db.get(QuestionRun, run_id).status == "cancelled"
            assert db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id)) is None
        login_as(client, "lifecycle-author")
        for path in (f"/api/question-runs/{run_id}", f"/api/paper-versions/{version_id}/file",
                     f"/api/question-runs/{run_id}/events?stream=false"):
            assert client.get(path).status_code == 404
        assert fake.roles == []
        return

    first_job = claim_job(sessions, settings)
    with sessions.begin() as db:
        db.get(QuestionRun, run_id).lease_until = now() - timedelta(seconds=1)
    recovered_job = claim_job(sessions, settings)
    assert recovered_job == ("run", run_id, first_job[2] + 1)
    process_job(sessions, settings, fake, first_job)
    assert fake.roles == []  # Superseded execution never reaches model inference.
    process_job(sessions, settings, fake, recovered_job)
    draft = client.get(f"/api/question-runs/{run_id}").json()
    assert draft["status"] == "review_pending" and draft["answer"] is None
    assert draft["answer_sha256"] is None
    progress = client.get(f"/api/question-runs/{run_id}/events?stream=false").json()
    assert "数据集 X 上 M 的 F1 为 82" not in str(progress)

    login_as(client, "lifecycle-reviewer")
    review_view = client.get(f"/api/question-runs/{run_id}").json()
    answer = review_view["answer"]
    assert answer["evidence_status"] == "supported", answer
    assert answer["claims"][0]["text"] == "数据集 X 上 M 的 F1 为 82。"
    source = answer["evidence"][0]
    assert source["paper_version_id"] == version_id
    assert source["sha256"] == hashlib.sha256(content).hexdigest()
    assert source["quote"] in content.decode()
    assert client.post(f"/api/answers/{answer['id']}/approval", json={
        "decision": "accept", "answer_sha256": review_view["answer_sha256"]}).status_code == 200

    login_as(client, "lifecycle-author")
    published = client.get(f"/api/question-runs/{run_id}").json()
    assert published["status"] == "completed" and published["answer"] == answer
    assert client.get(f"/api/evidence/{source['id']}").json() == source
    assert client.get(source["file_url"]).content == content
    with sessions() as db:
        assert len(list(db.scalars(select(AnswerRevision).where(AnswerRevision.run_id == run_id)))) == 1
    resumed = client.get(f"/api/question-runs/{run_id}/events?stream=false&after_seq={progress[-1]['seq']}").json()
    assert [event["kind"] for event in resumed] == ["completed"]


def create_pending_answer(client, sessions, settings, fake, question):
    key = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    response = client.post("/api/question-runs", headers={"Idempotency-Key": f"review-{key}-key"},
                           json={"snapshot_id": "snap1", "question": question})
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    job = claim_job(sessions, settings)
    assert job and job[1] == run_id
    from app.worker import process_job
    process_job(sessions, settings, fake, job)
    with sessions() as db:
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        evidence_id = db.scalar(select(EvidenceSpan.id).where(EvidenceSpan.run_id == run_id))
        return run_id, answer.id, evidence_id


def test_answer_review_hides_draft_until_editor_accepts(client, sessions, settings, fake):
    settings.answer_review_required = True
    assert client.post("/api/corpora/corpus1/members", json={
        "username": "author", "password": "test-password", "role": "reader"}).status_code == 201
    assert client.post("/api/corpora/corpus1/members", json={
        "username": "reviewer", "password": "test-password", "role": "editor"}).status_code == 201
    assert client.post("/api/corpora/corpus1/members", json={
        "username": "other-reader", "password": "test-password", "role": "reader"}).status_code == 201
    login_as(client, "author")
    run_id, answer_id, evidence_id = create_pending_answer(
        client, sessions, settings, fake, "M 的 F1 是多少？")

    draft = client.get(f"/api/question-runs/{run_id}").json()
    assert draft["status"] == "review_pending" and draft["answer"] is None and draft["state"] == {}
    assert draft["answer_sha256"] is None
    assert draft["is_owner"] is True
    assert client.get(f"/api/answers/{answer_id}").status_code == 404
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 404
    assert client.get(f"/api/question-runs/{run_id}/trace").status_code == 409

    login_as(client, "reviewer")
    queue = client.get("/api/corpora/corpus1/answer-reviews")
    assert queue.status_code == 200 and [item["run"]["id"] for item in queue.json()] == [run_id]
    assert queue.json()[0]["run"]["is_owner"] is False
    review_view = client.get(f"/api/question-runs/{run_id}").json()
    assert review_view["answer"]["id"] == answer_id and review_view["answer"]["claims"]
    assert review_view["is_owner"] is False
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 200
    assert client.get(f"/api/question-runs/{run_id}/events?stream=false").status_code == 404
    assert client.get(f"/api/question-runs/{run_id}/trace").status_code == 404
    assert client.get("/api/corpora/corpus1/question-runs").json() == []
    observed_hash = review_view["answer_sha256"]
    decision = client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "accept", "answer_sha256": observed_hash})
    assert decision.status_code == 200 and decision.json()["status"] == "completed"
    assert client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "accept", "answer_sha256": observed_hash}).status_code == 200
    assert client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "accept", "answer_sha256": "0" * 64}).status_code == 409
    assert client.get("/api/corpora/corpus1/answer-reviews").json() == []
    assert client.get(f"/api/question-runs/{run_id}").status_code == 404
    assert client.get(f"/api/question-runs/{run_id}/events?stream=false").status_code == 404
    assert client.get(f"/api/answers/{answer_id}").status_code == 200
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 200

    login_as(client, "other-reader")
    assert client.get(f"/api/question-runs/{run_id}").status_code == 404
    assert client.get(f"/api/answers/{answer_id}").status_code == 200
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 200

    login_as(client, "author")
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 200
    published = client.get(f"/api/question-runs/{run_id}").json()
    assert published["answer"]["id"] == answer_id
    assert published["is_owner"] is True
    assert client.get(f"/api/answers/{answer_id}").json()["id"] == answer_id
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 200
    with sessions() as db:
        approval = db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run_id))
        experience = db.scalar(select(AcceptedAnswerExperience).where(
            AcceptedAnswerExperience.run_id == run_id))
        assert approval.decision == "accept" and approval.answer_sha256 == experience.answer_sha256
        assert experience.answer_id == answer_id
    with sessions.begin() as db:
        claim = db.scalar(select(AnswerClaim).where(AnswerClaim.answer_id == answer_id))
        claim.text += "（被意外修改）"
    assert client.get(f"/api/question-runs/{run_id}").status_code == 409
    assert client.get(f"/api/answers/{answer_id}").status_code == 409


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_answer_approval_requires_the_content_the_reviewer_saw(client, sessions, settings, fake, decision):
    settings.answer_review_required = True
    run_id, answer_id, _ = create_pending_answer(client, sessions, settings, fake, "M 的 F1 是多少？")
    observed = client.get(f"/api/question-runs/{run_id}").json()
    assert observed["answer"]["id"] == answer_id
    assert client.get(f"/api/answers/{answer_id}/approval").json()["answer_sha256"] == observed["answer_sha256"]

    with sessions.begin() as db:
        claim = db.scalar(select(AnswerClaim).where(AnswerClaim.answer_id == answer_id))
        claim.text += "（内容已修订）"
    body = {"decision": decision, "answer_sha256": observed["answer_sha256"], "reason": "证据需要复核"}
    stale = client.post(f"/api/answers/{answer_id}/approval", json=body)
    assert stale.status_code == 409 and stale.json()["code"] == "answer_content_changed"
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "review_pending"
        assert db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run_id)) is None
        assert db.scalar(select(AcceptedAnswerExperience).where(AcceptedAnswerExperience.run_id == run_id)) is None

    refreshed = client.get(f"/api/question-runs/{run_id}").json()
    assert refreshed["answer_sha256"] != observed["answer_sha256"]
    assert "内容已修订" in refreshed["answer"]["claims"][0]["text"]
    body["answer_sha256"] = refreshed["answer_sha256"]
    accepted = client.post(f"/api/answers/{answer_id}/approval", json=body)
    assert accepted.status_code == 200
    assert accepted.json()["status"] == ("completed" if decision == "accept" else "rejected")
    assert client.post(f"/api/answers/{answer_id}/approval", json=body).json() == accepted.json()
    with sessions() as db:
        record = db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run_id))
        assert record.answer_sha256 == refreshed["answer_sha256"]
        experience = db.scalar(select(AcceptedAnswerExperience).where(AcceptedAnswerExperience.run_id == run_id))
        assert (experience is not None) == (decision == "accept")


def test_answer_approval_rejects_missing_or_invalid_content_hash(client):
    # Request validation must fail before looking up the supplied answer ID.
    for decision in ("accept", "reject"):
        for value in (None, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64):
            body = {"decision": decision, "reason": "证据需要复核"}
            if value is not None:
                body["answer_sha256"] = value
            response = client.post("/api/answers/unused-answer/approval", json=body)
            assert response.status_code == 422, response.text
            assert any(error["loc"] == ["body", "answer_sha256"] for error in response.json()["detail"])


def test_published_hash_survives_evidence_reordering_and_reads_legacy_records(
        client, sessions, settings, fake, monkeypatch):
    settings.answer_review_required = True
    run_id, answer_id, _ = create_pending_answer(client, sessions, settings, fake, "M 的 F1 是多少？")
    with sessions.begin() as db:
        claim = db.scalar(select(AnswerClaim).where(AnswerClaim.answer_id == answer_id))
        block = db.get(SourceBlock, "block1")
        db.add(EvidenceSpan(id="000-extra-evidence", run_id=run_id, block_id=block.id,
                            char_start=0, char_end=12, quote=block.text[:12]))
        db.flush()
        db.add(ClaimEvidence(id="extra-evidence-link", claim_id=claim.id,
                             evidence_id="000-extra-evidence", relation="supports"))
    observed = client.get(f"/api/question-runs/{run_id}").json()
    assert len(observed["answer"]["evidence"]) == 2
    assert client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "accept", "answer_sha256": observed["answer_sha256"]}).status_code == 200

    original_answer_data = services_module.answer_data

    def reordered_answer_data(*args, **kwargs):
        payload = deepcopy(original_answer_data(*args, **kwargs))
        payload["evidence"].reverse()
        for claim in payload["claims"]:
            claim["evidence"].reverse()
        return payload

    # Simulate an alternative SQL query order without changing any content.
    with monkeypatch.context() as patch:
        patch.setattr(services_module, "answer_data", reordered_answer_data)
        published = client.get(f"/api/question-runs/{run_id}")
        assert published.status_code == 200
        assert published.json()["answer_sha256"] == observed["answer_sha256"]

    legacy_hash = digest(observed["answer"])
    assert legacy_hash != answer_content_hash(observed["answer"])
    with sessions.begin() as db:
        db.scalar(select(AnswerApprovalRecord).where(
            AnswerApprovalRecord.run_id == run_id)).answer_sha256 = legacy_hash
        db.scalar(select(AcceptedAnswerExperience).where(
            AcceptedAnswerExperience.run_id == run_id)).answer_sha256 = legacy_hash
    assert client.get(f"/api/answers/{answer_id}").status_code == 200
    assert client.get(f"/api/question-runs/{run_id}").status_code == 200
    with sessions.begin() as db:
        db.scalar(select(AnswerClaim).where(AnswerClaim.answer_id == answer_id)).text += "（被修改）"
    response = client.get(f"/api/answers/{answer_id}")
    assert response.status_code == 409 and response.json()["code"] == "answer_approval_mismatch"



@pytest.mark.parametrize("change", ["revoke_author", "deactivate_version", "delete_document"])
def test_answer_approval_rechecks_author_and_snapshot(client, sessions, settings, fake, change):
    settings.answer_review_required = True
    author = client.post("/api/corpora/corpus1/members", json={
        "username": "author", "password": "test-password", "role": "reader"}).json()
    client.post("/api/corpora/corpus1/members", json={
        "username": "reviewer", "password": "test-password", "role": "editor"})
    login_as(client, "author")
    run_id, answer_id, _ = create_pending_answer(client, sessions, settings, fake, "M 的 F1 是多少？")
    login_as(client, "reviewer")
    observed_hash = client.get(f"/api/question-runs/{run_id}").json()["answer_sha256"]
    login_as(client, "admin")
    if change == "revoke_author":
        assert client.delete(f"/api/corpora/corpus1/members/{author['id']}").status_code == 200
    elif change == "delete_document":
        assert client.delete("/api/documents/paper1").status_code == 200
    else:
        with sessions.begin() as db:
            db.get(PaperVersion, "v1").is_active = False
    login_as(client, "reviewer")
    response = client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "accept", "answer_sha256": observed_hash})
    assert response.status_code == (404 if change == "revoke_author" else 409)
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "review_pending"
        assert db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run_id)) is None
        assert db.scalar(select(AcceptedAnswerExperience).where(AcceptedAnswerExperience.run_id == run_id)) is None
    # Rejection remains available so a reviewer can close an obsolete draft.
    rejected = client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "reject", "reason": "来源或权限已变化", "answer_sha256": observed_hash})
    assert rejected.status_code == 200 and rejected.json()["status"] == "rejected"


def test_answer_rejection_requires_reason_and_replay_carries_feedback(
        client, sessions, settings, fake, monkeypatch):
    from app import team_api

    settings.answer_review_required = True
    monkeypatch.setattr(team_api, "get_settings", lambda: settings)
    client.post("/api/corpora/corpus1/members", json={
        "username": "author", "password": "test-password", "role": "reader"})
    client.post("/api/corpora/corpus1/members", json={
        "username": "reviewer", "password": "test-password", "role": "editor"})
    login_as(client, "author")
    run_id, answer_id, _ = create_pending_answer(client, sessions, settings, fake, "延迟是多少？")
    login_as(client, "reviewer")
    observed_hash = client.get(f"/api/question-runs/{run_id}").json()["answer_sha256"]
    missing = client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "reject", "answer_sha256": observed_hash})
    assert missing.status_code == 422
    rejected = client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "reject", "reason": "需要补上对照方法。", "answer_sha256": observed_hash})
    assert rejected.status_code == 200 and rejected.json()["status"] == "rejected"
    assert client.get(f"/api/question-runs/{run_id}").status_code == 404
    assert client.post(f"/api/answers/{answer_id}/approval", json={
        "decision": "accept", "answer_sha256": observed_hash}).status_code == 409

    login_as(client, "author")
    run = client.get(f"/api/question-runs/{run_id}").json()
    assert run["answer"] is None
    assert run["answer_review"] == {"decision": "reject", "reason": "需要补上对照方法。"}
    assert client.get(f"/api/answers/{answer_id}").status_code == 404
    replay = client.post(f"/api/question-runs/{run_id}/replay", headers={
        "Idempotency-Key": "rejected-answer-replay-1"})
    assert replay.status_code == 202
    assert replay.json()["snapshot_id"] == run["snapshot_id"]
    assert "审核意见：需要补上对照方法。" in replay.json()["question"]
    assert replay.json()["id"] != run_id
    with sessions() as db:
        approval = db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run_id))
        assert approval.reason == "需要补上对照方法。"
        assert db.scalar(select(AcceptedAnswerExperience).where(
            AcceptedAnswerExperience.run_id == run_id)) is None


def test_markdown_and_word_sections_survive_chunking_without_invented_pages(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("# 入职\n\n先申请账号。\n\n## 发布\n\n发布需要两人复核。\n", encoding="utf-8")
    docx = tmp_path / "test.docx"
    doc = Document()
    doc.add_heading("权限申请", 1)
    doc.add_paragraph("负责人审批后开通只读账号。")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "角色", "阅读者"
    doc.save(docx)
    for path in (md, docx):
        raw, pages, warnings = extract_document(path, path.name)
        assert pages is None and warnings
        blocks = [SourceBlock(id=f"b{i}", parse_revision_id="p", **{k: v for k, v in b.items() if k != "page_height"})
                  for i, b in enumerate(raw)]
        chunks = build_chunks(blocks, {0: 842})
        for chunk in chunks:
            sections = {blocks[int(r["block_id"][1:])].section for r in chunk["source_ranges"]}
            assert len(sections) == 1
        assert any("账号" in b.text or "复核" in b.text for b in blocks)


def test_upload_all_formats_and_reject_binary_markdown(client):
    doc = Document()
    doc.add_paragraph("这是测试用团队说明，不是实际制度。")
    stream = BytesIO()
    doc.save(stream)
    assert client.post("/api/corpora/corpus1/papers", files={"file": ("test.docx", stream.getvalue())}).status_code == 202
    assert client.post("/api/corpora/corpus1/papers", files={"file": ("test.md", b"\x00broken")}).status_code == 400
    assert client.post("/api/corpora/corpus1/papers", files={"file": ("test.doc", b"legacy")}).status_code == 400


@pytest.mark.parametrize("filename,content", [
    ("guide.txt", b"UTF-8 text"),
    ("table.csv", b"name,role\nLin,editor\n"),
    ("table.tsv", b"name\trole\nLin\teditor\n"),
    ("data.json", b'{"role":"reader"}'),
    ("data.jsonl", b'{"role":"reader"}\n'),
    ("page.html", b"<p>visible content</p>"),
])
def test_upload_accepts_supported_data_formats(client, filename, content):
    response = client.post("/api/corpora/corpus1/papers", files={"file": (filename, content)})
    assert response.status_code == 202, response.text


def test_upload_accepts_real_xlsx_workbook(client):
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active.append(["姓名", "角色"])
    workbook.active.append(["Lin", "编辑者"])
    stream = BytesIO()
    workbook.save(stream)
    workbook.close()

    response = client.post("/api/corpora/corpus1/papers", files={
        "file": ("team.xlsx", stream.getvalue())})
    assert response.status_code == 202, response.text


def test_upload_rejects_unsupported_and_non_utf8_text_files(client):
    assert client.post("/api/corpora/corpus1/papers", files={
        "file": ("archive.zip", b"PK\x03\x04")}).status_code == 400
    assert client.post("/api/corpora/corpus1/papers", files={
        "file": ("invalid.json", b"\xff\xfe\x00\x01")}).status_code == 400


def test_upload_rechecks_target_document_after_reading_file(client, sessions, settings, monkeypatch):
    original_require_role = main_module.require_role
    role_checks = 0

    def delete_target_after_read(*args, **kwargs):
        nonlocal role_checks
        role = original_require_role(*args, **kwargs)
        role_checks += 1
        # This check follows the file read and the rollback of the initial
        # admission transaction. Simulate deletion by another request then.
        if role_checks == 2:
            with sessions.begin() as db:
                db.get(Paper, "paper1").deleted = True
        return role

    monkeypatch.setattr(main_module, "require_role", delete_target_after_read)
    content = b"Replacement uploaded while its document was removed."
    response = client.post("/api/corpora/corpus1/papers", data={"paper_id": "paper1"},
                           files={"file": ("replacement.txt", content)})
    assert response.status_code == 404, response.text
    with sessions() as db:
        versions = list(db.scalars(select(PaperVersion).where(PaperVersion.paper_id == "paper1")))
        assert [version.id for version in versions] == ["v1"]
        assert db.get(Paper, "paper1").deleted
    assert not list(settings.storage_dir.iterdir())


def test_upload_rejects_same_sha_across_documents_in_one_corpus(client):
    content = b"same knowledge base content"
    first = client.post("/api/corpora/corpus1/papers", files={"file": ("first.txt", content)})
    assert first.status_code == 202, first.text
    duplicate = client.post("/api/corpora/corpus1/papers", files={"file": ("second.txt", content)})
    assert duplicate.status_code == 409
    assert "知识库" in duplicate.json()["detail"]


def test_reranker_selects_actual_result_blocks_instead_of_repeated_headings(sessions, settings, corpus, fake):
    with sessions.begin() as db:
        for i in range(10):
            db.add(SourceBlock(id=f"header-{i}", parse_revision_id="parse1", page_index=0, block_index=i,
                               bbox=[], text=f"Background heading {i}\n"))
        db.add(SourceBlock(id="actual-result", parse_revision_id="parse1", page_index=5, block_index=0,
                           bbox=[], text="Method M latency 120 ms, baseline 100 ms.\n"))
        db.flush()
        chunk = db.get(Chunk, "chunk1")
        blocks = list(db.scalars(select(SourceBlock).where(SourceBlock.id.like("header-%"))))
        blocks.append(db.get(SourceBlock, "actual-result"))
        chunk.source_ranges = [{"block_id": b.id, "char_start": 0, "char_end": len(b.text)} for b in blocks]
    run_id = None
    with sessions() as db:
        run_id = create_run(db, RunCreate(snapshot_id="snap1", question="推理延迟是多少？"),
                            "rerank-test-1", "local-user", settings).id
    job = claim_job(sessions, settings)
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))
    state["plan"] = {"subquestions": [{"id": "q1", "queries": ["latency"]}]}
    original = fake.structured

    def structured(role, prompt, data, schema):
        if role == "Reranker":
            assert data["candidates"][0]["retrieved_for_subquestion_ids"] == ["q1"]
            return Reranking(items=[{"chunk_id": "chunk1", "score": 1,
                                     "relevant_block_ids": ["actual-result"]}]), {
                                         "input_tokens": 1, "output_tokens": 1, "elapsed_seconds": 0.}
        return original(role, prompt, data, schema)
    fake.structured = structured
    state = Workflow(sessions, settings, fake, run_id, job[2]).retrieve(state)
    assert [s["block_id"] for s in state["sources"]] == ["actual-result"]
    assert "120 ms" in state["sources"][0]["quote"]


def test_failed_update_leaves_old_version_available(sessions, settings, corpus):
    from app.worker import process_job
    from conftest import FakeProvider

    settings.storage_dir.mkdir()
    content = "# 更新草稿\n\n应保留原版直到新索引成功。".encode()
    (settings.storage_dir / "new.md").write_bytes(content)
    with sessions.begin() as db:
        db.add(PaperVersion(id="bad-update", paper_id="paper1", filename="new.md", storage_key="new.md",
                            sha256=hashlib.sha256(content).hexdigest(), byte_size=len(content),
                            status="uploaded", is_active=False))
    class BrokenEmbedding(FakeProvider):
        def embed(self, *args):
            from app.errors import ProviderError
            raise ProviderError("模拟向量服务失败")
    process_job(sessions, settings, BrokenEmbedding(), claim_job(sessions, settings))
    with sessions() as db:
        assert db.get(PaperVersion, "bad-update").status == "failed"
        assert db.get(PaperVersion, "v1").is_active
        assert create_snapshot(db, "corpus1", ["v1"], "local-user").manifest


def test_celery_task_runs_durable_job(sessions, settings, corpus, fake, monkeypatch):
    from app import db as database
    from app import providers, tasks

    settings.database_url = "sqlite:///test-only"
    monkeypatch.setattr(tasks, "settings", settings)
    monkeypatch.setattr(database, "Session", sessions)
    monkeypatch.setattr(providers, "ModelProvider", lambda _: fake)
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                         "celery-task-1", "local-user", settings)
        run_id = run.id
    tasks.execute_next.apply(throw=True)
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "completed"
