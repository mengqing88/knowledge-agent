import hashlib
import json
from types import SimpleNamespace

import pytest
from app.core.errors import DomainError

from evaluation.context_recall import (
    WideCandidates,
    query_facets,
    source_successors,
    validated_order,
    with_facet_context,
    with_forward_context,
)
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


def test_query_facets_keep_source_text_and_bounded_contrastive_clauses():
    assert query_facets("请说明管理员如何批准外部成员访问资料？") == []
    assert query_facets("管理员如何批准外部成员？撤权以后如何处理历史任务？") == [
        "管理员如何批准外部成员", "撤权以后如何处理历史任务"]
    assert query_facets("一方认为应保留原始记录，另一方认为需要脱敏以后共享。原文分别怎样表达？") == [
        "一方认为应保留原始记录", "另一方认为需要脱敏以后共享"]


def test_facet_context_spends_fixed_slots_on_unseen_continuation_and_stays_scoped():
    ranking = ["a", "b", "a0", "d", "b0", "f"]
    successors = {"a": "a0", "a0": "a1", "a1": "a2", "b": "b0", "b0": "b1",
                  "b1": "b2", "b2": "b3", "d": "d0", "p": "e", "e": "e0", "z": "z0"}
    allowed = set(ranking) | set(successors) | set(successors.values())
    result = with_facet_context(ranking, [["b", "a"], ["e", "a"]], successors, allowed)
    assert result == ["a", "a0", "b", "b0", "a1", "d", "d0", "b1", "e", "p", "b2", "b3"]
    assert len(result) == len(set(result)) == 12
    assert "z" not in result and "z0" not in result
    assert with_facet_context(ranking, [], successors, allowed) == with_forward_context(
        ranking, successors, allowed)[:12]
    for parts in ([[]], [["outside"]], [["a", "a"]], [["a"], ["b"], ["d"]]):
        with pytest.raises(ValueError):
            with_facet_context(ranking, parts, successors, allowed)


def test_facet_context_uses_predecessor_only_for_a_new_facet_leader():
    ranking = [f"full{i}" for i in range(12)]
    successors = {"full0": "full0next", "left": "leader", "leader": "right", "right": "farther"}
    allowed = set(ranking) | set(successors) | set(successors.values())
    result = with_facet_context(ranking, [["leader"]], successors, allowed)
    assert result[:8] == ["full0", "full0next", "full1", "full2", "full3", "full4", "leader", "left"]
    assert len(result) == len(set(result)) == 12
    with pytest.raises(ValueError, match="nonbranching"):
        with_facet_context(ranking, [["leader"]], {**successors, "other": "leader"},
                           allowed | {"other"})


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


def test_facet_benchmark_freezes_all_inputs_and_rejects_incomplete_scores(sessions, corpus, tmp_path, monkeypatch):
    from scripts.benchmarks import benchmark_facet_recall as facet_benchmark

    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    suite_path, vectors = tmp_path / "suite.json", tmp_path / "vectors.json"
    query = "管理员怎样核对原始资料？审核完成以后怎样共享？"
    suite_path.write_text(json.dumps({"snapshot_id": "snap1", "questions": [
        {"id": "q1", "question": query, "gold": [{"block_id": "block1", "char_start": 0, "char_end": 12}]}]}),
        encoding="utf-8")
    vectors.write_text(json.dumps({"identity": "frozen", "vectors": {
        hashlib.sha256(query.encode()).hexdigest(): [1., 0., 0.]}}), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "download-manifest.json").write_text(json.dumps({"model": "test", "revision": "frozen", "files": {
        "config.json": {"bytes": 2, "sha256": benchmark.sha(model / "config.json")}}}), encoding="utf-8")
    base_run = tmp_path / "tmp/base"
    benchmark.prepare(SimpleNamespace(database=tmp_path / "test.db", suite=suite_path, query_vectors=vectors,
                                      model=model, output=base_run))
    original = json.loads((base_run / "requests.jsonl").read_text(encoding="utf-8"))
    metadata = {"model_manifest_sha256": benchmark.sha(model / "download-manifest.json"),
                "seconds": .1, "truncated_pairs": 0}
    (base_run / "scores.jsonl").write_text(json.dumps({**metadata, "id": "q1",
        "request_sha256": request_digest(original), "scores": [{"id": "chunk1", "score": 1.}]}), encoding="utf-8")
    benchmark.score_run(SimpleNamespace(run=base_run, scores=base_run / "scores.jsonl"))
    facet_run = tmp_path / "tmp/facet"
    facet_benchmark.prepare(SimpleNamespace(base_run=base_run, model=model, output=facet_run))
    with pytest.raises(ValueError, match="immutable"):
        facet_benchmark.prepare(SimpleNamespace(base_run=base_run, model=model, output=facet_run))
    requests = [json.loads(line) for line in (facet_run / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(requests) == 2 and all(set(r) == {"id", "question", "candidates"} for r in requests)
    results = [{**metadata, "id": r["id"], "request_sha256": request_digest(r),
                "scores": [{"id": "chunk1", "score": 1.}]} for r in requests]
    scores = facet_run / "scores.jsonl"
    scores.write_text(json.dumps(results[0]), encoding="utf-8")
    args = SimpleNamespace(run=facet_run, scores=scores)
    with pytest.raises(ValueError, match="exactly once"):
        facet_benchmark.score_run(args)
    assert not (facet_run / "report.json").exists()
    scores.write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")
    facet_benchmark.score_run(args)
    report = json.loads((facet_run / "report.json").read_text(encoding="utf-8"))
    assert report["threshold"]["met"] and report["summary_at_12"]["covered"] == 1
    assert report["new_paid_model_requests"] == 0 and report["human_reviewed"] is False

    child_run = tmp_path / "tmp/facet-child"
    facet_benchmark.prepare(SimpleNamespace(base_run=base_run, model=model, output=child_run,
                                            reference_run=facet_run, target_recall=.95))
    child_scores = child_run / "scores.jsonl"
    child_scores.write_bytes(scores.read_bytes())
    facet_benchmark.score_run(SimpleNamespace(run=child_run, scores=child_scores))
    child_report = json.loads((child_run / "report.json").read_text(encoding="utf-8"))
    assert child_report["baseline_at_12"]["covered"] == 1
    assert child_report["threshold"]["target"] == .95
    assert child_report["inference_reused"] and child_report["new_local_inference_pairs"] == 0
