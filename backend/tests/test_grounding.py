from copy import deepcopy

import pytest
from app.grounding import condition_present, numeric_issues
from app.models import AnswerClaim, AnswerRevision, QuestionRun
from app.schemas import RunCreate
from app.services import claim_job, create_run
from app.worker import process_job
from conftest import FakeProvider
from sqlalchemy import select


def fact(value="82", subject="M", metric="F1", unit="number", evidence_id="e1"):
    return {"evidence_id": evidence_id, "quote": f"{subject} achieved {metric} {value}",
            "subject": subject, "metric": metric, "value": value, "unit": unit,
            "conditions": ["dataset X"]}


def claim(text="M 的 F1 为 82。", value="82", unit="number", facts=None, operation="reported"):
    facts = facts or [fact()]
    return {"id": "c1", "text": text, "conditions": ["数据集 X"], "subquestion_ids": ["q1"],
            "evidence": [{"evidence_id": f["evidence_id"], "relation": "supports"} for f in facts],
            "numeric_checks": [{"statement": text, "value": value, "unit": unit,
                                "operation": operation, "facts": facts}]}


SOURCES = {"e1": {"quote": "On dataset X, M achieved F1 82 and baseline achieved F1 80."}}


def test_original_value_with_source_binding_passes():
    assert numeric_issues(claim(), SOURCES) == []


def test_verbatim_quote_may_include_the_final_sentence_period():
    draft = claim()
    draft["numeric_checks"][0]["facts"][0]["quote"] = "On dataset X, M achieved F1 82."
    assert numeric_issues(draft, {"e1": {"quote": "On dataset X, M achieved F1 82."}}) == []


@pytest.mark.parametrize("change", ["invented", "metric", "unit", "swap", "no_check", "extra", "cropped"])
def test_numeric_failures_despite_real_existing_citation(change):
    draft = claim()
    sources = deepcopy(SOURCES)
    check = draft["numeric_checks"][0]
    if change == "invented":
        draft["text"] = check["statement"] = "M 的 F1 为 99。"
        check["value"] = "99"
    elif change == "metric":
        draft["text"] = check["statement"] = "M 的准确率为 82。"
    elif change == "unit":
        draft["text"] = check["statement"] = "M 的 F1 为 82%。"
        check["unit"] = "%"
    elif change == "swap":
        draft["text"] = check["statement"] = "M 的 F1 为 80。"
        check["value"] = "80"
        check["facts"] = [fact(value="80", subject="baseline")]
    elif change == "no_check":
        draft["numeric_checks"] = []
    elif change == "extra":
        draft["text"] += " 延迟为 120 ms。"
    else:
        sources["e1"]["quote"] = "On dataset X, M achieved F1 82%."
    issues = numeric_issues(draft, sources)
    assert issues and all(i["type"] == "numeric_mismatch" for i in issues)


def test_compound_statement_cannot_hide_swapped_pairs():
    draft = claim("M 的 F1 为 80，baseline 的 F1 为 82。")
    assert numeric_issues(draft, SOURCES)


def test_wrong_dataset_and_extra_repeated_value_are_not_covered():
    draft = claim("数据集 Y 上 M 的 F1 为 82。")
    assert numeric_issues(draft, SOURCES)
    draft = claim()
    draft["text"] += "准确率也为 82。"
    assert numeric_issues(draft, SOURCES)


def test_explicit_shared_value_is_not_mistaken_for_a_numeric_swap():
    sources = {"e1": {"quote": "On dataset X, M and baseline both achieved F1 75."}}
    binding = fact("75")
    binding["quote"] = "M and baseline both achieved F1 75"
    assert not numeric_issues(claim("M 的 F1 为 75。", "75", facts=[binding]), sources)


@pytest.mark.parametrize("text", ["M 的 F1 为八十二分。", "M 的 F1 为 1e3。", "M 的 F1 为 1,000。"])
def test_unsupported_number_formats_require_revision(text):
    assert numeric_issues(claim(text), SOURCES)


def test_ms_to_seconds_conversion_is_computed():
    source = {"e1": {"quote": "On dataset X, M achieved latency 120 ms."}}
    binding = fact("120", metric="latency", unit="ms")
    assert not numeric_issues(claim("M 的延迟为 0.12 s。", "0.12", "s", [binding]), source)
    assert numeric_issues(claim("M 的延迟为 120 s。", "120", "s", [binding]), source)


@pytest.mark.parametrize("operation,text,value,unit,valid", [
    ("difference", "M 的 F1 82 与 baseline 的 F1 80 相减，差为 2。", "2", "number", True),
    ("difference", "M 的 F1 82 与 baseline 的 F1 80 相减，差为 3。", "3", "number", False),
    ("relative_change", "M 的 F1 82 对 baseline 的 F1 80，相对变化为 2.5%。", "2.5", "%", True),
    ("relative_change", "M 的 F1 82 对 baseline 的 F1 80，相对变化为 2%。", "2", "%", False),
])
def test_derived_values_use_server_arithmetic(operation, text, value, unit, valid):
    facts = [fact(), fact("80", subject="baseline")]
    assert bool(numeric_issues(claim(text, value, unit, facts, operation), SOURCES)) is not valid


def test_derived_statement_cannot_swap_entities_while_hiding_valid_facts():
    facts = [fact(), fact("80", subject="baseline")]
    draft = claim("M 的 F1 80 与 baseline 的 F1 82 相减，差为 2。", "2", "number", facts, "difference")
    assert numeric_issues(draft, SOURCES)


def test_percentage_difference_requires_percentage_points():
    sources = {"e1": {"quote": "On dataset X, M achieved accuracy 82% and baseline achieved accuracy 80%."}}
    facts = [fact(metric="accuracy", unit="%"), fact("80", "baseline", "accuracy", "%")]
    valid = claim("M 的准确率 82% 与 baseline 的准确率 80% 相减，差为 2 个百分点。", "2", "pp", facts, "difference")
    assert numeric_issues(valid, sources) == []
    invalid = claim("M 的准确率 82% 与 baseline 的准确率 80% 相减，差为 2%。", "2", "%", facts, "difference")
    assert numeric_issues(invalid, sources)


def test_cross_dataset_comparison_cannot_hide_conditions():
    sources = {"e1": {"quote": "On dataset X under matched setting, M achieved F1 82."},
               "e2": {"quote": "On dataset Y under matched setting, baseline achieved F1 80."}}
    facts = [fact(), fact("80", "baseline", evidence_id="e2")]
    for item in facts:
        item["conditions"] = ["matched setting"]
    draft = claim("M 的 F1 82 与 baseline 的 F1 80 相减，差为 2。", "2", "number", facts, "difference")
    assert any(i["type"] == "comparison_mismatch" for i in numeric_issues(draft, sources))


def test_unsupported_units_and_unbound_conditions_do_not_pass():
    draft = claim()
    draft["numeric_checks"][0]["facts"][0]["unit"] = "kg"
    assert numeric_issues(draft, SOURCES)
    draft = claim()
    draft["numeric_checks"][0]["facts"][0]["conditions"] = ["dataset Y"]
    assert numeric_issues(draft, SOURCES)
    draft = claim()
    draft["conditions"] = []
    assert numeric_issues(draft, SOURCES)


def test_numeric_condition_cannot_borrow_an_equal_metric_value():
    draft = claim()
    draft["conditions"].append("温度 82")
    assert any(issue["type"] == "numeric_mismatch" for issue in numeric_issues(draft, SOURCES))


@pytest.mark.parametrize("source_setting,valid", [
    ("setting A", True), ("settingA", True), ("setting AB", False),
    ("setting A1", False), ("presetting A", False),
    ("setting A-1", False), ("setting A.1", False), ("pre-setting A", False),
])
def test_numeric_fact_condition_requires_complete_identifier(source_setting, valid):
    binding = fact()
    binding["conditions"] = ["setting A"]
    draft = claim(facts=[binding])
    draft["conditions"] = ["setting A"]
    source = {"e1": {"quote": f"On {source_setting}, M achieved F1 82."}}
    assert bool(numeric_issues(draft, source)) is not valid


@pytest.mark.parametrize("text,valid", [
    ("setting A.", True), ("setting A-1", False), ("setting A.1", False),
    ("pre-setting A", False), ("setting A, next", True),
])
def test_condition_identifier_punctuation_boundaries(text, valid):
    assert condition_present(text, "setting A") is valid


class AdversarialProvider(FakeProvider):
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario

    def structured(self, role, prompt, data, schema):
        result, usage = super().structured(role, prompt, data, schema)
        output = result.model_dump()
        if role == "Planner" and self.scenario in {"uncovered", "explicit_gap", "removed"}:
            output["subquestions"].append({"id": "q2", "target": "延迟", "queries": ["M inference latency"]})
        if role == "Analyst":
            candidate = output["claims"][0]
            if self.scenario == "swapped":
                candidate["text"] = "M 的 F1 为 80。"
                candidate["numeric_checks"] = [{"statement": candidate["text"], "operation": "reported",
                    "value": "80", "unit": "number", "facts": [fact("80", "baseline",
                    evidence_id=data["sources"][0]["id"])]}]
                candidate["numeric_checks"][0]["facts"][0]["quote"] = "baseline F1 80"
            if self.scenario == "invented_numeric_condition":
                candidate["conditions"].append("温度 82")
            if self.scenario == "unknown_id":
                candidate["subquestion_ids"] = ["invented-q"]
            if self.scenario == "explicit_gap":
                output["subquestion_gaps"] = [{"subquestion_id": "q2", "reason": "当前材料未报告延迟。"}]
            if self.scenario == "removed":
                second = deepcopy(candidate)
                second["id"], second["subquestion_ids"] = "c2", ["q2"]
                second["text"], second["numeric_checks"] = "M 的推理速度更快。", []
                output["claims"].append(second)
        if role == "Verifier":
            if self.scenario == "uncertain":
                output["claim_reviews"][0]["verdict"] = "uncertain"
            if self.scenario == "no_review":
                output["claim_reviews"] = []
            if self.scenario == "bad_review_evidence":
                output["claim_reviews"][0]["evidence_ids"] = ["invented"]
            if self.scenario == "removed":
                output["issues"].append({"claim_id": "c2", "type": "unsupported_claim", "message": "未报告延迟"})
        return schema.model_validate(output), usage


@pytest.mark.parametrize("mode", ["baseline_rag", "without_verifier", "full_agent"])
def test_program_rejects_swapped_numbers_even_when_verifier_approves(sessions, settings, corpus, mode):
    with sessions() as db:
        run_id = create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1？", mode=mode),
                            "numeric", "local-user", settings).id
    process_job(sessions, settings, AdversarialProvider("swapped"), claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert answer.evidence_status == "insufficient"
        assert any(i["type"] == "numeric_mismatch" for i in answer.issues)
        assert not list(db.scalars(select(AnswerClaim).where(AnswerClaim.answer_id == answer.id)))


@pytest.mark.parametrize("scenario", ["uncovered", "explicit_gap", "removed"])
def test_missing_subquestion_remains_visible_after_filtering(sessions, settings, corpus, scenario):
    with sessions() as db:
        run_id = create_run(db, RunCreate(snapshot_id="snap1", question="F1 和延迟分别如何？"),
                            "coverage", "local-user", settings).id
    process_job(sessions, settings, AdversarialProvider(scenario), claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert answer.evidence_status == "partial"
        assert any("q2" in gap and "延迟" in gap for gap in answer.gaps)
        assert run.state["coverage"][0]["subquestion_id"] == "q2"


@pytest.mark.parametrize("scenario", ["uncertain", "no_review", "bad_review_evidence", "unknown_id",
                                      "invented_numeric_condition"])
def test_uncertain_or_uninspectable_claim_is_not_published(sessions, settings, corpus, scenario):
    with sessions() as db:
        run_id = create_run(db, RunCreate(snapshot_id="snap1", question="F1 是多少？"),
                            "review", "local-user", settings).id
    process_job(sessions, settings, AdversarialProvider(scenario), claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert answer.evidence_status == "insufficient"
