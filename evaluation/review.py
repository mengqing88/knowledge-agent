"""Export pending human reviews and score only complete, result-bound annotations."""
import argparse
import hashlib
import json
import re
from pathlib import Path

MODES = ("baseline_rag", "full_agent", "without_verifier")


def modes_for(manifest):
    modes = manifest.get("mode_order", MODES)
    if (not isinstance(modes, (list, tuple)) or not modes
            or any(not isinstance(mode, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", mode) for mode in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError("评测模式必须是无重复的安全目录名")
    return modes


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load_records(directory, questions):
    records = {}
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = row["question_id"]
        if key not in questions or key in records:
            raise ValueError("评测结果包含未知或重复 question_id")
        records[key] = (row, sha256(path))
    return records


def links_for(row):
    run = row.get("run", {})
    if row.get("technical_error") or run.get("status") != "completed":
        return []
    return [(c["id"], e["evidence_id"], e["relation"])
            for c in (run.get("answer") or {}).get("claims", []) for e in c["evidence"]]


def required_keys(question):
    keys = [c["key"] for c in question.get("required_claims", [])]
    if len(set(keys)) != len(keys):
        raise ValueError("题集中的应答要点 key 重复")
    return set(keys)


def make_template(root, review_source="human"):
    if review_source not in ("human", "agent"):
        raise ValueError("Unknown review source")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = {q["question_id"]: q for q in manifest["questions"]}
    entries = []
    for mode in modes_for(manifest):
        records = load_records(root / mode, questions)
        for key, question in questions.items():
            row, result_hash = records.get(key, ({}, None))
            entries.append({"mode": mode, "question_id": key, "result_sha256": result_hash,
                            "reviewer": "", "independent_human_review": False,
                            "links": [{"claim_id": c, "evidence_id": e, "relation": r,
                                       "supports_claim": None, "relation_correct": None}
                                      for c, e, r in links_for(row)],
                            "covered_required_keys": None, "forbidden_conclusions_present": None,
                            "notes": "待人工逐条阅读原文、完整答案及错误/缺失记录后填写。"})
            required_keys(question)
    if review_source == "agent":
        for entry in entries:
            entry.update(review_source="agent", agent_review=False, action_correct=None,
                         conditions_correct=None, notes="待 Agent 对照冻结原文核对答案、条件、引用和遗漏。")
        return {"version": 1, "review_source": "agent", "manifest_sha256": sha256(manifest_path),
                "note": "Agent 审核；不是独立人工审核或正式业务准确率，不改变答案发布审批。", "reviews": entries}
    return {"version": 1, "manifest_sha256": sha256(manifest_path),
            "note": "人工独立判定；不要复制 Verifier 的 approved 作为标注。空值不是零分或通过。",
            "reviews": entries}


def summarize_manual(root, mode, annotation=None):
    return _summarize(root, mode, annotation, "human")


def summarize_agent(root, mode, annotation=None):
    return _summarize(root, mode, annotation, "agent")


def _summarize(root, mode, annotation, source):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = {q["question_id"]: q for q in manifest["questions"]}
    records = load_records(root / mode, questions)
    result = {"status": "pending_independent_human_review", "questions": len(questions), "reviewed_questions": 0,
              "citation_semantic_support_rate": None, "relation_correct_rate": None,
              "required_point_coverage": None, "forbidden_conclusion_question_rate": None,
              "support_links": 0, "all_links": 0, "required_points": sum(len(required_keys(q)) for q in questions.values()),
              "fictional": any(q.get("fictional") for q in questions.values()), "dev_only": manifest.get("dev_only", True)}
    if source == "agent":
        result.update(status="pending_agent_review", review_source="agent", independent_human_review=False,
                      action_correct_rate=None, conditions_correct_rate=None)
    if annotation is None or (source == "human" and annotation.get("review_source") == "agent"):
        return result
    if source == "agent" and annotation.get("review_source") != "agent":
        raise ValueError("Agent 评分需要显式 Agent 标注")
    if annotation.get("version") != 1 or annotation.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("人工标注版本或 manifest SHA 不匹配")
    reviews, seen = {}, set()
    for review in annotation["reviews"]:
        identity = (review["mode"], review["question_id"])
        if identity in seen or identity[0] not in modes_for(manifest) or identity[1] not in questions:
            raise ValueError("人工标注包含未知或重复的模式/题号")
        seen.add(identity)
        if identity[0] == mode:
            reviews[identity[1]] = review
    supported, relations, covered, forbidden, actions, conditions = 0, 0, 0, 0, 0, 0
    for key, question in questions.items():
        review = reviews.get(key)
        row, result_hash = records.get(key, ({}, None))
        if not review:
            continue
        if review.get("result_sha256") != result_hash:
            raise ValueError("人工标注与当前逐题结果 SHA 不匹配；请重新审阅")
        if source == "human":
            if review.get("independent_human_review") is not True:
                continue
        else:
            if review.get("agent_review") is not True:
                continue
            if review.get("review_source") != "agent" or review.get("independent_human_review") is not False:
                raise ValueError("Agent 标注不能冒充人工审核")
            if not review.get("notes", "").strip():
                raise ValueError("Agent 必须记录逐题审核依据")
            if any(type(review.get(k)) is not bool for k in ("action_correct", "conditions_correct")):
                raise ValueError("Agent 必须核对动作与适用条件")
            actions += review["action_correct"]
            conditions += review["conditions_correct"]
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            raise ValueError("独立人工标注必须填写审核者")
        expected_links = links_for(row)
        judged = [(link["claim_id"], link["evidence_id"], link["relation"]) for link in review["links"]]
        if len(set(judged)) != len(judged) or set(judged) != set(expected_links):
            raise ValueError("人工标注必须覆盖且只能覆盖本答案全部引用关系")
        for link in review["links"]:
            if type(link.get("relation_correct")) is not bool:
                raise ValueError("引用关系判定尚未完成")
            relations += link["relation_correct"]
            result["all_links"] += 1
            if link["relation"] == "supports":
                if type(link.get("supports_claim")) is not bool:
                    raise ValueError("支持引用的语义判定尚未完成")
                result["support_links"] += 1
                supported += link["supports_claim"]
        keys = review.get("covered_required_keys")
        if (not isinstance(keys, list) or any(not isinstance(k, str) for k in keys) or
                len(set(keys)) != len(keys) or not set(keys) <= required_keys(question)):
            raise ValueError("覆盖要点必须是本题已定义 key 的无重复列表")
        run = row.get("run", {})
        if (row.get("technical_error") or run.get("status") != "completed") and keys:
            raise ValueError("失败、取消或缺失答案的要点覆盖必须为零")
        if type(review.get("forbidden_conclusions_present")) is not bool:
            raise ValueError("禁用结论检查尚未完成")
        covered += len(keys)
        forbidden += review["forbidden_conclusions_present"]
        result["reviewed_questions"] += 1
    if result["reviewed_questions"] == len(questions) and questions:
        result.update(status="independently_annotated", covered_points=covered,
                      citation_semantic_support_rate=supported / result["support_links"] if result["support_links"] else None,
                      relation_correct_rate=relations / result["all_links"] if result["all_links"] else None,
                      required_point_coverage=covered / result["required_points"] if result["required_points"] else None,
                      forbidden_conclusion_question_rate=forbidden / len(questions))
        if source == "agent":
            result.update(status="agent_annotated", action_correct_rate=actions / len(questions),
                          conditions_correct_rate=conditions / len(questions))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-source", choices=["human", "agent"], default="human")
    args = parser.parse_args()
    # Never overwrite existing review work.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(make_template(args.run_directory, args.review_source), stream, ensure_ascii=False, indent=2)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
