import json
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest
from app.errors import BudgetExhausted

from evaluation.budget import BudgetLedger, BudgetProvider
from evaluation.review import make_template, summarize_agent, summarize_manual
from evaluation.run import case_order, select_questions
from scripts.compare_saved_reranker import ranking_metrics, saved_cases
from scripts.run_public_evaluation import seed_budget_ledger


def prices():
    return {"chat": {"input": "2", "output": "8", "model": "m", "base_url": "https://chat.test/v1",
                     "source": "frozen-test-price"},
            "embedding": {"input": "0.7", "output": "0", "model": "e", "base_url": "https://embed.test/v1",
                          "source": "frozen-test-price"}}


def reserve(ledger, inp=100, out=100):
    return ledger.reserve("chat", "https://chat.test/v1", "m", inp, out)


def test_global_budget_reserves_before_send_and_keeps_margin(tmp_path):
    ledger = BudgetLedger(tmp_path / "cost.json", prices())
    ticket = reserve(ledger, 250000, 250000)
    assert json.loads(ledger.path.read_text())["calls"][ticket]["status"] == "reserved"
    assert ledger.total() == 2.5
    with pytest.raises(BudgetExhausted):
        reserve(ledger)
    ledger.settle(ticket, {"usage": {"input_tokens": 10, "output_tokens": 20}})
    assert ledger.total() < .001
    reserve(ledger)


def test_reconciled_prior_charge_is_added_to_the_inherited_budget(tmp_path):
    source = tmp_path / "prior.json"
    prior = BudgetLedger(source, prices())
    ticket = reserve(prior, 1000, 100)
    prior.settle(ticket, {"usage": {"input_tokens": 10, "output_tokens": 20}})
    destination = tmp_path / "continued" / "cost.json"
    assert seed_budget_ledger(source, destination, "0.25") == "0.25"
    continued = BudgetLedger(destination, prices())
    assert continued.total() == prior.total() + Decimal("0.25")
    assert prior.total() < .001
    with pytest.raises(BudgetExhausted):
        continued.reserve("chat", "https://chat.test/v1", "m", 249000, 250000)


def test_budget_ledger_seed_rejects_unresolved_requests(tmp_path):
    source = tmp_path / "prior.json"
    prior = BudgetLedger(source, prices())
    reserve(prior)
    with pytest.raises(ValueError, match="unresolved"):
        seed_budget_ledger(source, tmp_path / "continued.json", 0)


def test_budget_ledger_seed_rejects_non_cny_ledger(tmp_path):
    source = tmp_path / "prior.json"
    prior = BudgetLedger(source, prices())
    prior.data["currency"] = "USD"
    prior.save()
    with pytest.raises(ValueError, match="CNY"):
        seed_budget_ledger(source, tmp_path / "continued.json", 0)


def test_selected_question_subset_is_deterministic_and_validated():
    questions = [{"question_id": "q01"}, {"question_id": "q04"}, {"question_id": "q07"}]
    assert select_questions(questions, ["q07", "q04"]) == questions[1:]
    assert select_questions(questions, None) == questions
    with pytest.raises(ValueError, match="Unknown"):
        select_questions(questions, ["q99"])
    with pytest.raises(ValueError, match="Duplicate"):
        select_questions(questions, ["q04", "q04"])


def test_saved_reranker_replay_preserves_candidate_text_and_maps_gold_docs(tmp_path):
    (tmp_path / "full_agent").mkdir()
    question = {"question_id": "q1", "required_claims": [{"evidence": [{"document": "doc-a"}]}]}
    saved = {"question_id": "q1", "trace": {
        "retrievals": [{"rankings": [{"chunk_id": "c1", "paper_version_id": "doc-b"},
                                      {"chunk_id": "c2", "paper_version_id": "doc-a"}]}],
        "model_calls": [{"role": "Reranker", "request": {"data": {
            "question": "Q", "plan": {"subquestions": [{"target": "target"}]},
            "candidates": [{"chunk_id": "c1", "blocks": [{"text": "irrelevant"}]},
                           {"chunk_id": "c2", "blocks": [{"text": "evidence"}]}]}},
                         "response": {"items": [{"chunk_id": "c2"}, {"chunk_id": "c1"}]}}]}}
    (tmp_path / "full_agent/0001.json").write_text(json.dumps(saved), encoding="utf-8")
    cases, excluded = saved_cases(tmp_path, {"questions": [question]})
    assert not excluded
    assert cases[0]["candidates"] == [{"id": "c1", "text": "irrelevant"}, {"id": "c2", "text": "evidence"}]
    assert cases[0]["question"] == "Q\n\n问题要点：\n- target"
    assert ranking_metrics(cases[0]["llm_order"], cases[0]["candidate_documents"], ["doc-a"])["mrr"] == 1


def test_ranking_metrics_are_document_level_and_exclude_unanswerable():
    mapping = {"first": "doc-a", "duplicate": "doc-a", "second": "doc-b", "wrong": "doc-c"}
    metrics = ranking_metrics(["first", "duplicate", "second", "wrong"], mapping, ["doc-b", "doc-c"])
    assert metrics == pytest.approx({"recall_at_3": 1.0, "mrr": .5, "ndcg_at_3": 0.693426})
    assert ranking_metrics(["wrong"], mapping, []) is None


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 1}, {"input_tokens": True, "output_tokens": 1}])
def test_unknown_usage_retains_reservation_and_stops_every_following_request(tmp_path, usage):
    ledger = BudgetLedger(tmp_path / "cost.json", prices())
    ticket = reserve(ledger)
    ledger.settle(ticket, {"usage": usage})
    assert str(ledger.total()) == "0.001"
    with pytest.raises(BudgetExhausted):
        reserve(ledger)


def test_restart_cannot_erase_unconfirmed_call_or_reset_prices(tmp_path):
    path = tmp_path / "cost.json"
    reserve(BudgetLedger(path, prices()))
    resumed = BudgetLedger(path, prices())
    with pytest.raises(BudgetExhausted):
        reserve(resumed)
    changed = prices()
    changed["chat"]["input"] = "1"
    with pytest.raises(ValueError):
        BudgetLedger(path, changed)


def test_unpriced_endpoint_and_exceeded_usage_stop(tmp_path):
    ledger = BudgetLedger(tmp_path / "cost.json", prices())
    with pytest.raises(ValueError, match="Unpriced"):
        ledger.reserve("chat", "https://other.test", "m", 10, 10)
    ticket = reserve(ledger)
    ledger.settle(ticket, {"usage": {"input_tokens": 101, "output_tokens": 1}})
    with pytest.raises(BudgetExhausted):
        reserve(ledger)


def test_provider_uses_single_global_ledger_and_exact_embedding_cache(tmp_path, settings, monkeypatch):
    from app.providers import ModelProvider
    settings = settings.model_copy(update={"provider_max_retries": 0, "embedding_model": "e",
                                          "embedding_base_url": "https://embed.test/v1"})
    ledger = BudgetLedger(tmp_path / "cost.json", prices())
    sent = []

    def post(self, base, key, endpoint, payload):
        assert ledger.data["calls"][-1]["status"] == "reserved"
        sent.append(payload)
        return {"data": [{"index": i, "embedding": [1., 0., 0.]} for i in range(len(payload["input"]))],
                "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    monkeypatch.setattr(ModelProvider, "_post", post)
    shared = {}
    a, b = BudgetProvider(settings, ledger, shared), BudgetProvider(settings, ledger, shared)
    assert a.embed(["same question"], 3) == b.embed(["same question"], 3)
    assert len(sent) == len(ledger.data["calls"]) == 1
    assert b.last_embedding_usage["cache_hit"] and b.last_embedding_usage["input_tokens"] == 0


def test_agent_review_is_hash_bound_and_never_counts_as_human(tmp_path):
    (tmp_path / "full_agent").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"mode_order": ["full_agent"], "questions": [
        {"question_id": "q1", "required_claims": [{"key": "fact"}]},
        {"question_id": "q2", "required_claims": [{"key": "missing"}]}]}))
    path = tmp_path / "full_agent/0001.json"
    path.write_text(json.dumps({"question_id": "q1", "run": {"status": "completed", "answer": {"claims": []}}}))
    template = make_template(tmp_path, "agent")
    for entry in template["reviews"]:
        entry.update(agent_review=True, reviewer="Codex-test", action_correct=True, conditions_correct=True,
                     notes="Read frozen source and checked omissions; absent q2 retained in denominator",
                     covered_required_keys=["fact"] if entry["question_id"] == "q1" else [],
                     forbidden_conclusions_present=False)
    assert summarize_agent(tmp_path, "full_agent", template)["required_point_coverage"] == .5
    assert summarize_manual(tmp_path, "full_agent", template)["reviewed_questions"] == 0
    template["reviews"][0]["independent_human_review"] = True
    with pytest.raises(ValueError, match="冒充"):
        summarize_agent(tmp_path, "full_agent", template)
    template["reviews"][0]["independent_human_review"] = False
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA"):
        summarize_agent(tmp_path, "full_agent", template)


def test_two_arms_are_interleaved_once():
    questions = [{"question_id": str(n)} for n in range(12)]
    order = case_order(questions, ["baseline_rag", "full_agent"])
    assert len(order) == 24
    assert [(n, mode) for n, _, mode in order][:4] == [
        (1, "baseline_rag"), (1, "full_agent"), (2, "baseline_rag"), (2, "full_agent")]


def test_process_identity_rejects_pid_reuse_and_legacy_kill():
    import os
    if os.name != "nt":
        pytest.skip("Windows process records")
    root = Path(__file__).resolve().parents[2]
    command = r"""
    param($Root)
    . (Join-Path $Root 'scripts/process-state.ps1')
    $script:fake = $null
    function Get-CimInstance { param($ClassName,$Filter,$ErrorAction) return $script:fake }
    $created = [datetime]::UtcNow
    $record = [pscustomobject]@{pid=123; creation_utc=$created.ToUniversalTime().ToString('o'); command_line='manage.py api'}
    $states = @()
    $states += Get-ProjectProcessState $record $Root
    $script:fake = [pscustomobject]@{ExecutablePath='unrelated.exe';CommandLine='other';CreationDate=$created}
    $states += Get-ProjectProcessState $record $Root
    $script:fake.ExecutablePath = Join-Path $Root '.venv\Scripts\python.exe'
    $script:fake.CommandLine = 'manage.py api'
    $states += Get-ProjectProcessState 123 $Root
    $states += Get-ProjectProcessState $record $Root
    $record.creation_utc = $created.AddSeconds(-1).ToUniversalTime().ToString('o')
    $states += Get-ProjectProcessState $record $Root
    $states | ConvertTo-Json -Compress
    """
    result = subprocess.run(["powershell", "-NoProfile", "-Command", "& {" + command + "} '" + str(root) + "'"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    assert json.loads(result.stdout) == ["dead", "unrelated", "unverified", "owned", "unrelated"]
