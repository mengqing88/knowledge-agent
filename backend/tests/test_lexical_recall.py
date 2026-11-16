import json
from types import SimpleNamespace

import pytest
from app.core.models import Chunk, IndexRevision
from app.knowledge.lexical import LexicalBM25, lexical_tokens, lexical_windows

from evaluation.lexical_recall import round_robin
from scripts.benchmarks import benchmark_lexical_recall as benchmark


def test_chinese_paraphrase_has_lexical_overlap_without_whole_sentence_match():
    index = LexicalBM25({"evidence": "外部成员需要管理员批准后才有访问权限。",
                         "distractor": "管理员今天讨论了食堂菜单。"})
    assert index.search("外部成员怎样申请访问权限？")[0] == "evidence"
    assert index.search("nonexistentxyz") == []
    tokens = lexical_tokens("访问，权限 service.timeout_seconds ≤ 100")
    assert {"访", "问", "权", "限", "访问", "权限"} <= set(tokens)
    assert "问权" not in tokens
    assert "service.timeout_seconds" in tokens and "100" in tokens
    assert "service" not in tokens and "timeout" not in tokens


def test_bm25_rare_terms_beat_repeated_common_word_and_ties_are_stable():
    docs = {f"common-{i}": "common filler" for i in range(8)}
    docs.update(rare="needle answer", spam="common " * 200)
    assert LexicalBM25(docs).search("needle common")[0] == "rare"
    assert LexicalBM25({"b": "same text", "a": "same text"}).search("same") == ["a", "b"]
    assert LexicalBM25({}).search("query") == []
    with pytest.raises(ValueError):
        LexicalBM25(docs).search("query", True)


def test_window_and_balanced_controls_return_only_real_source_content_and_ids():
    source = "甲" * 250 + "。下一句话说明条件。\n最后一句。"
    windows = lexical_windows(source)
    assert windows and all(window in source and len(window) <= 200 for window in windows)
    assert any("下一句话说明条件。" in window for window in windows)
    assert round_robin([["a", "b", "c"], ["b", "d"]]) == ["a", "b", "d", "c"]


def test_score_requires_complete_source_range_in_the_correct_block():
    chunks = {"partial": SimpleNamespace(source_ranges=[{"block_id": "b1", "char_start": 0, "char_end": 5}]),
              "wrong": SimpleNamespace(source_ranges=[{"block_id": "b2", "char_start": 0, "char_end": 20}]),
              "correct": SimpleNamespace(source_ranges=[{"block_id": "b1", "char_start": 0, "char_end": 20}])}
    label = [{"block_id": "b1", "char_start": 3, "char_end": 12}]
    assert benchmark.score(["partial", "wrong"], chunks, label)["covered"] == 0
    assert benchmark.score(["correct"], chunks, label)["covered"] == 1


def test_offline_benchmark_keeps_snapshot_scope_and_never_needs_a_provider(
        sessions, corpus, tmp_path, monkeypatch):
    with sessions.begin() as db:
        db.add(IndexRevision(id="outside-index", parse_revision_id="parse1", config_version="outside",
                             embedding_model="test-embedding", embedding_dimensions=3, status="ready"))
        db.flush()
        db.add(Chunk(id="outside-secret", index_revision_id="outside-index", text="M F1 dataset X " * 10,
                     source_ranges=[], embedding=[1., 0., 0.]))
    suite = {"snapshot_id": "snap1", "original_suite_sha256": "frozen-source",
             "questions": [{"id": "q1", "question": "M F1 dataset X", "gold": [
                 {"block_id": "block1", "char_start": 0, "char_end": 12}]}]}
    saved = {"status": "completed", "snapshot_id": "snap1", "question_set_sha256": "frozen-source",
             "cases": [{"id": "q1", "profiles": {"vector": {"top_30": ["chunk1"]},
                                                     "rrf": {"top_30": ["chunk1"]}}}]}
    suite_path, saved_path = tmp_path / "suite.json", tmp_path / "saved.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    saved_path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    args = SimpleNamespace(database=tmp_path / "test.db", suite=suite_path, saved_rankings=saved_path,
                           clause_vectors=None, output=tmp_path / "tmp/result.json")
    benchmark.run(args)
    result = json.loads(args.output.read_text(encoding="utf-8"))
    assert result["status"] == "completed" and result["new_model_calls"] == 0
    assert result["parameters"]["han_ngram_sizes"] == [1, 2]
    assert result["chunk_count"] == 1 and result["oracle_coverable_labels"] == 1
    for method in result["cases"][0]["methods"].values():
        assert method["top30"] == ["chunk1"]
        assert method["scores"]["5"]["complete"]
    with pytest.raises(ValueError, match="immutable"):
        benchmark.run(args)


def test_offline_benchmark_rejects_forged_source_offsets(sessions, corpus):
    from app.core.models import Snapshot

    with sessions() as db:
        snapshot = db.get(Snapshot, "snap1")
        for label in ({"block_id": "unknown", "char_start": 0, "char_end": 10},
                      {"block_id": "block1", "char_start": False, "char_end": 10},
                      {"block_id": "block1", "char_start": 0, "char_end": 9999}):
            with pytest.raises(ValueError, match="invalid or outside"):
                benchmark.validate_scope(db, snapshot, [{"question": "test", "gold": [label]}])
