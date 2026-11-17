import pytest
from app.core.config import Settings
from app.core.db import Base, make_engine
from app.core.models import (
    Chunk,
    Corpus,
    IndexRevision,
    Paper,
    PaperVersion,
    ParseRevision,
    Snapshot,
    SourceBlock,
    User,
)
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def sessions(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    Base.metadata.create_all(engine)
    yield sessionmaker(engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, model="test-model", embedding_model="test-embedding",
                    storage_dir=tmp_path / "pdfs", max_run_seconds=60, answer_review_required=False,
                    query_rewrite_enabled=False, retrieval_strategy="legacy")


@pytest.fixture
def corpus(sessions, settings):
    from app.knowledge.ingestion import index_config
    with sessions.begin() as db:
        from app.core.auth import hash_password
        db.add(User(id="local-user", username="admin", password_hash=hash_password("test-password")))
        db.flush()
        db.add(Corpus(id="corpus1", owner_id="local-user", name="教学样例"))
        db.flush()
        db.add(Paper(id="paper1", corpus_id="corpus1", title="Fictional paper A"))
        db.flush()
        db.add(PaperVersion(id="v1", paper_id="paper1", sha256="a"*64, storage_key="a.pdf",
                            filename="a.pdf", byte_size=1, status="ready"))
        db.flush()
        db.add(ParseRevision(id="parse1", paper_version_id="v1", parser="test", config_version="test", status="ready"))
        db.flush()
        db.add(SourceBlock(id="block1", parse_revision_id="parse1", page_index=3, block_index=0,
                           bbox=[72, 100, 500, 150],
                           text="On dataset X, M achieved F1 82 and baseline F1 80.\nLatency was not measured.\n"))
        db.add(IndexRevision(id="index1", parse_revision_id="parse1", config_version=index_config(settings),
                             embedding_model="test-embedding", embedding_dimensions=3, status="ready"))
        db.flush()
        db.add(Chunk(id="chunk1", index_revision_id="index1", text="M achieved F1 82 on dataset X.",
                     source_ranges=[{"block_id": "block1", "char_start": 0, "char_end": 72}],
                     embedding=[1., 0., 0.]))
        db.add(Snapshot(id="snap1", corpus_id="corpus1", owner_id="local-user", manifest=[{
            "paper_id": "paper1", "title": "Fictional paper A", "paper_version_id": "v1", "sha256": "a"*64,
            "parse_revision_id": "parse1", "index_revision_id": "index1",
            "embedding_model": "test-embedding", "embedding_dimensions": 3,
        }]))
    return "corpus1"


class FakeProvider:
    """Deterministic control-flow test double. Never selected by normal application settings."""
    def __init__(self, *, invalid=False, reject=False, clarify=False, gaps=False, fail=False):
        self.invalid, self.reject, self.clarify, self.gaps, self.fail = invalid, reject, clarify, gaps, fail
        self.roles = []

    def embed(self, texts, expected_dim=0):
        return [[1., 0., 0.] for _ in texts]

    def structured(self, role, prompt, data, schema):
        from app.core.errors import ProviderError
        self.roles.append(role)
        if self.fail:
            raise ProviderError("模型服务超时")
        if role == "Planner":
            output = {"normalized_question": data["question"], "needs_clarification": self.clarify,
                      "clarification": "请明确比较的方法。" if self.clarify else None,
                      "subquestions": [] if self.clarify else [
                          {"id": "q1", "target": "F1", "queries": ["M F1 dataset X"]}]}
        elif role == "Reranker":
            output = {"items": [{"chunk_id": c["chunk_id"], "score": .9,
                                 "relevant_block_ids": [b["block_id"] for b in c["blocks"]]}
                                for c in data["candidates"]]}
        elif role == "EvidenceGate":
            output = {"reviews": [{"subquestion_id": q["id"], "status": "ready",
                       "evidence_ids": [data["sources"][0]["id"]], "reason": "原文包含问题所需事实",
                       "followup_queries": []} for q in data["plan"]["subquestions"]]}
        elif role == "Analyst":
            output = {"claims": [{"id": "c1", "text": "数据集 X 上 M 的 F1 为 82。",
                                   "conditions": ["数据集 X"], "subquestion_ids": ["q1"],
                                   "numeric_checks": [{"statement": "数据集 X 上 M 的 F1 为 82。",
                                       "operation": "reported", "value": "82", "unit": "number", "facts": [{
                                           "evidence_id": "invented" if self.invalid else data["sources"][0]["id"],
                                           "quote": "M achieved F1 82", "subject": "M", "metric": "F1",
                                           "value": "82", "unit": "number", "conditions": ["dataset X"]}]}],
                                   "evidence": [{
                                       "evidence_id": "invented" if self.invalid else data["sources"][0]["id"],
                                       "relation": "supports"}]}],
                      "gaps": ["当前检索未找到显存依据。"] if self.gaps else [], "has_conflict": False}
        else:
            output = {"approved_claim_ids": [] if self.reject else ["c1"],
                      "claim_reviews": [{"claim_id": "c1", "verdict": "rejected" if self.reject else "approved",
                                         "evidence_ids": [data["draft"]["claims"][0]["evidence"][0]["evidence_id"]],
                                         "rationale": "原文明确给出数据集 X 的 M F1 数值。"}],
                      "issues": ([{"claim_id": "c1", "type": "unsupported_claim", "message": "原文不支持"}]
                                 if self.reject else []),
                      "followup_queries": ["M GPU memory footprint"] if self.gaps else []}
        return schema.model_validate(output), {"input_tokens": 10, "output_tokens": 10, "elapsed_seconds": .001}


@pytest.fixture
def fake():
    return FakeProvider()
