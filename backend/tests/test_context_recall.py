import hashlib
import json
from types import SimpleNamespace

import pytest
from app.core.errors import DomainError

from evaluation.context_recall import WideCandidates, source_successors, validated_order, with_forward_context
from scripts.benchmarks import benchmark_context_recall as benchmark
from scripts.models.local_rerank import request_digest


def test_forward_context_consumes_result_slots_and_follows_ranking():
    ranked = [f"c{i}" for i in range(20)]
    successors = {f"c{i}": f"n{i}" for i in range(10)}
    allowed = set(ranked) | set(successors.values())
    result = with_forward_context(ranked, successors, allowed)
    assert result[:12] == [item for i in range(6) for item in (f"c{i}", f"n{i}")]
    assert len(result) == len(set(result))
    assert set(result) <= allowed
    assert with_forward_context(["c1", "c2"], {"c1": "c2"}, {"c1", "c2"}) == ["c1", "c2"]


@pytest.mark.parametrize("ranking,successors", [(["outside"], {}), (["a", "a"], {}),
                                                 (["a"], {"a": "outside"})])
def test_forward_context_rejects_foreign_or_duplicate_ids(ranking, successors):
    with pytest.raises(ValueError, match="outside scope"):
        with_forward_context(ranking, successors, {"a"})


def test_successors_use_source_order_and_never_cross_index_boundaries():
    def chunk(index, block, start):
        return SimpleNamespace(index_revision_id=index,
                               source_ranges=[{"block_id": block, "char_start": start, "char_end": start+1}])

    blocks = {"b1": SimpleNamespace(parse_revision_id="p1", page_index=1, block_index=0, text="abcdef"),
              "b2": SimpleNamespace(parse_revision_id="p1", page_index=2, block_index=0, text="abcdef")}
    chunks = {"uuid-a": chunk("i1", "b2", 0), "uuid-z": chunk("i1", "b1", 0),
              "uuid-m": chunk("i1", "b1", 4), "other-index": chunk("i2", "b1", 0)}
    assert source_successors(chunks, blocks, {"i1": "p1", "i2": "p1"}) == {
        "uuid-z": "uuid-m", "uuid-m": "uuid-a"}
    blocks["b2"].parse_revision_id = "outside"
    with pytest.raises(ValueError, match="outside its scoped"):
        source_successors(chunks, blocks, {"i1": "p1", "i2": "p1"})
    blocks["b2"].parse_revision_id = "p1"
    chunks["uuid-z"].source_ranges[0]["char_end"] = 999
    with pytest.raises(DomainError):
        source_successors(chunks, blocks, {"i1": "p1", "i2": "p1"})


def test_wide_candidates_only_use_passed_text_and_bounded_routes():
    index = WideCandidates({"needle": "外部成员须管理员批准后获得访问权限。", "menu": "今天食堂菜单"})
    routes, ranking = index.rankings("外部成员怎样申请访问权限？", ["menu", "needle"], limit=1)
    assert routes["full"] == ["needle"] and routes["window"] == ["needle"]
    assert routes["dense"] == ["menu"] and ranking == ["needle", "menu"]
    assert index.rankings("unknownxyz", [], limit=1)[1] == []
    with pytest.raises(ValueError):
        index.rankings("q", [], limit=True)


def test_inference_is_bound_to_full_question_text_candidate_order_and_text():
    request = {"id": "q", "question": "target?", "candidates": [{"id": "a", "text": "real evidence"}]}
    digest = request_digest(request)
    good = {"id": "q", "request_sha256": digest, "scores": [{"id": "a", "score": -.3}]}
    assert validated_order(good, request, digest) == ["a"]
    assert request_digest({**request, "question": "changed"}) != digest
    assert request_digest({**request, "candidates": [{"id": "a", "text": "changed"}]}) != digest
    with pytest.raises(ValueError, match="identity"):
        validated_order({**good, "request_sha256": "forged"}, request, digest)
    for bad_scores in ([], [{"id": "foreign", "score": 1.}], [{"id": "a", "score": float("nan")}],
                       [{"id": "a", "score": True}]):
        with pytest.raises(ValueError):
            validated_order({**good, "scores": bad_scores}, request, digest)


def test_offline_prepare_and_score_keep_input_scope_and_all_questions(sessions, corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    suite = {"snapshot_id": "snap1", "questions": [{"id": "q1", "question": "M F1 dataset X", "gold": [
        {"block_id": "block1", "char_start": 0, "char_end": 12}]}]}
    suite_path, vectors = tmp_path / "suite.json", tmp_path / "vectors.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    key = hashlib.sha256(suite["questions"][0]["question"].encode()).hexdigest()
    vectors.write_text(json.dumps({"identity": "frozen", "vectors": {key: [1., 0., 0.]}}), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    model_manifest = {"model": "test", "revision": "frozen", "files": {"config.json": {
        "bytes": 2, "sha256": benchmark.sha(model / "config.json")}}}
    (model / "download-manifest.json").write_text(json.dumps(model_manifest), encoding="utf-8")
    args = SimpleNamespace(database=tmp_path / "test.db", suite=suite_path, query_vectors=vectors,
                           model=model, output=tmp_path / "tmp/experiment")
    benchmark.prepare(args)
    with pytest.raises(ValueError, match="immutable"):
        benchmark.prepare(args)
    requests = [json.loads(line) for line in (args.output / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(requests) == 1 and set(requests[0]) == {"id", "question", "candidates"}
    assert [c["id"] for c in requests[0]["candidates"]] == ["chunk1"]
    score_path = args.output / "scores.jsonl"
    result = {"id": "q1", "request_sha256": request_digest(requests[0]), "seconds": .2, "truncated_pairs": 0,
              "model_manifest_sha256": benchmark.sha(model / "download-manifest.json"),
              "scores": [{"id": "chunk1", "score": 1.}]}
    score_path.write_text(json.dumps({**result, "scores": []}) + "\n", encoding="utf-8")
    scoring = SimpleNamespace(run=args.output, scores=score_path)
    with pytest.raises(ValueError, match="omitted"):
        benchmark.score_run(scoring)
    assert not (args.output / "report.json").exists()
    score_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    benchmark.score_run(scoring)
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    assert report["threshold"]["met"] and report["threshold"]["total"] == 1
    assert report["new_paid_model_requests"] == 0 and not report["human_reviewed"]
    with pytest.raises(ValueError, match="immutable"):
        benchmark.score_run(scoring)
