import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.chunking import token_count
from app.errors import BudgetExhausted

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("chunk_benchmark", ROOT / "scripts/benchmark_chunking.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_retrieval_metric_requires_whole_gold_span_and_does_not_count_overlap_twice():
    gold = [{"block_id": "b", "char_start": 10, "char_end": 30},
            {"block_id": "c", "char_start": 0, "char_end": 4}]
    rows = [SimpleNamespace(text="partial", source_ranges=[{"block_id": "b", "char_start": 0, "char_end": 20}]),
            SimpleNamespace(text="whole", source_ranges=[{"block_id": "b", "char_start": 0, "char_end": 40}]),
            SimpleNamespace(text="repeat", source_ranges=[{"block_id": "b", "char_start": 0, "char_end": 40}])]
    result = benchmark.score_query(rows, gold)
    assert result["evidence_recall_at_1"] == 0
    assert result["evidence_recall_at_5"] == .5
    assert result["reciprocal_rank_at_5"] == .5
    assert benchmark.score_query([], gold)["hit_at_5"] == 0


def test_frozen_corpus_labels_are_unique_and_include_long_sources():
    manifest, documents, questions, blocks = benchmark.load_corpus(ROOT / "fixtures/chunking/manifest.json")
    assert manifest["fictional"] and not manifest["human_reviewed"]
    assert len(documents) == 6 and len(questions) == 15
    assert sum(row["split"] == "held_out" for row in questions) == 6
    assert max(token_count(block.text) for block in blocks) > 1024
    assert all(row["gold"] for row in questions)


def test_embedding_cache_reuses_queries_and_refuses_spend_beyond_limit(settings):
    provider = benchmark.CachedEmbedding(settings, max_tokens=6, max_texts=2)
    calls = []
    provider.provider.embed = lambda texts, dim: calls.append(texts) or [[1., 0.] for _ in texts]
    provider.provider.last_embedding_usage = {"usage_known": False}
    assert provider.embed(["one", "two"]) == [[1., 0.], [1., 0.]]
    assert provider.embed(["one"]) == [[1., 0.]]
    with pytest.raises(BudgetExhausted):
        provider.embed(["three"])
    assert len(calls) == 1


def test_resume_rejects_different_corpus_before_creating_external_resources(tmp_path, settings, monkeypatch):
    import json

    settings.embedding_base_url, settings.embedding_api_key = "https://example.invalid", "test-only"
    monkeypatch.setattr(benchmark, "Settings", lambda **_kwargs: settings)
    monkeypatch.setattr(benchmark, "scratch", lambda *_args: pytest.fail("must not create external resources"))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"corpus_manifest_sha256": "wrong"}), encoding="utf-8")
    args = SimpleNamespace(manifest=ROOT / "fixtures/chunking/manifest.json", max_embedding_tokens=100,
                           max_embedding_texts=10, cache_dir=None, resume_from=previous)
    with pytest.raises(ValueError, match="续跑语料"):
        benchmark.run(args, tmp_path / "output.json")
