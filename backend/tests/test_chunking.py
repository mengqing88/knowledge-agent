import hashlib
from types import SimpleNamespace

import pytest
from app.chunking import ChunkPolicy, clean, token_count
from app.config import Settings
from app.errors import BudgetExhausted, ProviderError
from app.ingestion import build_chunks, index_config, ingest_version
from app.models import Chunk, IndexRevision, PaperVersion, Snapshot, SourceBlock
from pydantic import ValidationError
from sqlalchemy import select


def block(text, key="source", page=0, section="正文"):
    return SourceBlock(id=key, parse_revision_id="parse", text=text, bbox=[],
                       page_index=page, block_index=0, section=section)


@pytest.mark.parametrize("strategy", ["fixed", "paragraph", "structure", "semantic"])
def test_chunks_enforce_reference_token_limit_cover_original_and_preserve_unicode(strategy):
    source = block("这是较长的中文段落，包含权限、预算与🧪测试。" * 95 + " tail " * 80)
    provider = SimpleNamespace(embed=lambda texts, dim=0: [[1., 0.] for _ in texts])
    chunks = build_chunks([source], {0: 842}, policy=ChunkPolicy(strategy, 64, 8),
                          provider=provider, model_name="test")
    assert len(chunks) > 2
    covered = set()
    for chunk in chunks:
        assert token_count(chunk["text"]) <= 64
        assert "\ufffd" not in chunk["text"]
        for span in chunk["source_ranges"]:
            assert span["block_id"] == source.id
            assert 0 <= span["char_start"] < span["char_end"] <= len(source.text)
            covered.update(range(span["char_start"], span["char_end"]))
    assert all(i in covered for i, char in enumerate(source.text) if not char.isspace())


def test_structure_prefers_sentence_boundaries_and_fixed_overlap_has_real_source_overlap():
    source = block("Alpha has a source and a condition. Beta preserves the original evidence. " * 30)
    structured = build_chunks([source], {}, policy=ChunkPolicy("structure", 64, 0))
    assert all(chunk["text"].endswith(".") for chunk in structured)
    fixed = build_chunks([source], {}, policy=ChunkPolicy("fixed", 64, 8))
    assert any(a["source_ranges"][-1]["char_end"] > b["source_ranges"][0]["char_start"]
               for a, b in zip(fixed, fixed[1:]))
    assert all(token_count(chunk["text"]) <= 64 for chunk in fixed)


def test_boundaries_and_searchable_json_path_do_not_change_citation_offsets():
    sources = [block("45", "json", section="service.timeout_seconds"),
               block("姓名: Lin | 角色: editor", "row", section="工作表：成员 · 第 2 行"),
               block("Other page.", "page", page=1)]
    chunks = build_chunks(sources, {}, policy=ChunkPolicy("structure", 64, 8))
    assert len(chunks) == 3
    assert "service.timeout_seconds" in chunks[0]["text"]
    for chunk, source in zip(chunks, sources):
        assert chunk["source_ranges"] == [{"block_id": source.id, "char_start": 0, "char_end": len(source.text)}]
        assert clean(source.text) in chunk["text"]


def test_selected_fixed_policy_keeps_opposite_page_conditions_separate():
    allowed = block("Access is allowed for operators only.", key="allowed", page=0, section="Access")
    denied = block("Access is not allowed for visitors.", key="denied", page=1, section="Access")

    chunks = build_chunks([allowed, denied], {}, policy=ChunkPolicy())

    assert len(chunks) == 2
    assert "allowed for operators" in chunks[0]["text"]
    assert "not allowed for visitors" in chunks[1]["text"]
    assert chunks[0]["source_ranges"] == [{"block_id": "allowed", "char_start": 0,
                                             "char_end": len(allowed.text)}]
    assert chunks[1]["source_ranges"] == [{"block_id": "denied", "char_start": 0,
                                             "char_end": len(denied.text)}]


def test_semantic_topic_break_uses_project_provider_and_does_not_overlap_unrelated_topics():
    source = block("Alpha topic. Beta topic. Gamma topic.")
    calls = []
    provider = SimpleNamespace(embed=lambda texts, dim=0: (
        calls.append(texts) or [[1., 0.], [1., 0.], [0., 1.]]))
    chunks = build_chunks([source], {}, policy=ChunkPolicy("semantic", 64, 8, .55),
                          provider=provider, model_name="test")
    assert len(calls) == 1 and len(chunks) == 2
    assert "Alpha" in chunks[0]["text"] and "Gamma" not in chunks[0]["text"]
    assert chunks[1]["text"] == "Gamma topic."


def test_semantic_unit_cap_fails_before_embedding():
    source = block("Alpha topic. Beta topic. Gamma topic.")
    calls = []
    provider = SimpleNamespace(embed=lambda texts, dim=0: calls.append(texts) or [[1., 0.] for _ in texts])
    policy = ChunkPolicy("semantic", 64, 8, .55, semantic_max_units=2)

    with pytest.raises(ProviderError, match="句段数超过配置上限"):
        build_chunks([source], {}, policy=policy, provider=provider, model_name="test")

    assert calls == []


def test_chunking_checks_deadline_during_long_source():
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == 8:
            raise BudgetExhausted("ingestion_time")

    with pytest.raises(BudgetExhausted):
        build_chunks([block("not one huge unchecked operation " * 3000)], {}, guard=guard)


def test_chunk_policy_is_validated_and_every_setting_changes_index_identity(settings):
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size_tokens=64, chunk_overlap_tokens=64)
    baseline = index_config(settings)
    for field, value in [("chunk_strategy", "structure"), ("chunk_size_tokens", 256),
                         ("chunk_overlap_tokens", 0), ("chunk_semantic_threshold", .8),
                         ("chunk_semantic_max_units", 64)]:
        assert index_config(settings.model_copy(update={field: value})) != baseline


def test_runtime_default_matches_the_selected_evaluated_profile(settings):
    selected = ChunkPolicy("fixed", 512, 64)
    assert ChunkPolicy.from_settings(settings) == selected == ChunkPolicy()
    source = block("审批条件和有效期限都应保留。" * 200)
    assert build_chunks([source], {}) == build_chunks([source], {}, policy=selected)


def test_changed_policy_builds_new_index_and_retains_old_chunks_and_snapshot(sessions, settings, corpus, fake):
    settings.storage_dir.mkdir()
    path = settings.storage_dir / "a.pdf"
    # The fixture version's filename can change for this isolated test only.
    path.write_text("# Guide\n" + "Access requires an editor approval. " * 180, encoding="utf-8")
    with sessions.begin() as db:
        version = db.get(PaperVersion, "v1")
        version.filename, version.status, version.attempt = "guide.md", "parsing", 1
        version.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        version.byte_size = path.stat().st_size
    ingest_version(sessions, settings, fake, "v1", 1)
    with sessions() as db:
        old_ids = set(db.scalars(select(Chunk.id)))
        old_manifest = db.get(Snapshot, "snap1").manifest
        old_indexes = set(db.scalars(select(IndexRevision.id)))
    changed = settings.model_copy(update={"chunk_size_tokens": 128, "chunk_overlap_tokens": 16})
    with sessions.begin() as db:
        version = db.get(PaperVersion, "v1")
        version.status, version.attempt = "parsing", 2
    ingest_version(sessions, changed, fake, "v1", 2)
    with sessions() as db:
        assert old_ids < set(db.scalars(select(Chunk.id)))
        assert len(set(db.scalars(select(IndexRevision.id))) - old_indexes) == 1
        assert db.get(Snapshot, "snap1").manifest == old_manifest
