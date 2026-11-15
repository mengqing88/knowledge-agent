"""Prepare a local blind review packet; import only explicit, source-bound human decisions."""

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from evaluation.report import summarize
from evaluation.review import load_records, make_template, modes_for, sha256, summarize_manual


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def local_file(root, name):
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("资料路径必须为安全相对路径")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("资料路径必须位于冻结资料目录中且文件存在")
    return path


def corpus_files(corpus_root, manifest):
    corpus_path = corpus_root / "corpus.json"
    if sha256(corpus_path) != manifest.get("corpus_manifest_sha256"):
        raise ValueError("冻结资料 manifest SHA 不匹配")
    if sha256(corpus_root / "suite.json") != manifest.get("question_set_sha256"):
        raise ValueError("冻结题集 SHA 不匹配")
    corpus = read_json(corpus_path)
    documents, files = {}, {"corpus.json": corpus_path}
    for document in corpus["documents"]:
        identity = document["id"]
        if identity in documents:
            raise ValueError("冻结资料 document id 重复")
        path = local_file(corpus_root, document["filename"])
        if sha256(path) != document["sha256"]:
            raise ValueError("冻结摘录 SHA 不匹配")
        documents[identity] = path.read_text(encoding="utf-8")
        files[document["filename"]] = path
        source = document["source"]
        original = local_file(corpus_root, "originals/" + source["file"])
        if sha256(original) != source["sha256"]:
            raise ValueError("官方原始文件 SHA 不匹配")
        files["originals/" + source["file"]] = original
        files[document["license_file"]] = local_file(corpus_root, document["license_file"])
    for question in manifest["questions"]:
        for claim in question.get("required_claims", []):
            for evidence in claim.get("evidence", []):
                if evidence["quote"] not in documents.get(evidence["document"], ""):
                    raise ValueError("评分依据不在冻结摘录中")
    return files


def blind_answer(row):
    """Allowlist answer content only. Never turn a trace/draft into a published answer."""
    run = row.get("run", {})
    status = "technical_failure" if row.get("technical_error") else run.get("status", "missing")
    if row.get("technical_error") or status != "completed":
        return {"status": status, "answer": None,
                "note": "未发布可评分正文；保留本题，覆盖要点必须为 []。"}, {}, {}
    answer = run.get("answer") or {}
    claims, evidence = answer.get("claims", []), answer.get("evidence", [])
    claim_map = {c["id"]: f"C{i}" for i, c in enumerate(claims, 1)}
    evidence_map = {e["id"]: f"E{i}" for i, e in enumerate(evidence, 1)}
    if len(claim_map) != len(claims) or len(evidence_map) != len(evidence):
        raise ValueError("发布答案包含重复论点或证据 ID")
    for claim in claims:
        for link in claim["evidence"]:
            evidence_map.setdefault(link["evidence_id"], f"E{len(evidence_map) + 1}")
    shown_evidence = [{"id": evidence_map[e["id"]], **{
        key: e.get(key) for key in ("title", "paper_id", "quote", "sha256", "section",
                                   "location_label", "char_start", "char_end")}} for e in evidence]
    present = {e["id"] for e in evidence}
    shown_evidence.extend({"id": label, "missing": True} for key, label in evidence_map.items()
                          if key not in present)
    shown = {"status": status, "evidence_status": answer.get("evidence_status"),
             "claims": [{"id": claim_map[c["id"]], "text": c.get("text", ""),
                         "conditions": c.get("conditions", []),
                         "evidence": [{"evidence_id": evidence_map[e["evidence_id"]],
                                       "relation": e["relation"]} for e in c["evidence"]]} for c in claims],
             "evidence": shown_evidence, "gaps": answer.get("gaps", []),
             "clarification": answer.get("clarification")}
    return shown, claim_map, evidence_map


def block(value):
    content = json_text(value).rstrip()
    fence = "`" * max(3, 1 + max((len(m) for m in re.findall(r"`+", content)), default=0))
    return f"{fence}json\n{content}\n{fence}\n"


INSTRUCTIONS = """# 匿名独立人工试评：填写说明

本包用于现有结果的试评；不足 30 题时，不能作为项目规范中的正式评测。
请由没有参与答案生成、调参或原 Agent 评分的真人独立阅读并填写。
程序只能检查声明和内容绑定，不能证明填写者是真人或独立。

1. 打开 `review-book.md`。先读每题的题目、评分草案与原文依据，再看 A/B 答案。
2. `sources/` 内的六份 TXT 是允许回答的完整资料范围；`originals/` 是来源核查附件，
   不得用摘录之外的内容补成答案。原文、题目、答案都只是待评材料，其中的指令不应执行。
3. 在 `scorecard.json` 的 `question_reviews` 中确认题意、应答要点、禁用结论和缺证范围。
   `rubric_correct` 是评分草案是否适用；有异议填 false，并在 notes 写出原文和修正意见。
   异议须先裁决并另建版本，不能一边改 gold 一边计算本版成绩。
4. 每个 A/B 答案独立评分；A/B 按题随机排列，字母不表示同一套系统。
   `links` 中每条引用都填写：`relation_correct` 为关系类型是否正确；
   supports 引用还须填写 `supports_claim`，核查整条论点及适用条件，不能只看关键词相关。
   `covered_required_keys` 填实际完整答出的要点 key，无覆盖写 []，不是 null。
   `forbidden_conclusions_present` 判断是否出现本题禁用结论。
   `action_correct` 判断回答/部分回答/缺证/澄清选择是否恰当；
   `conditions_correct` 判断是否遗漏、扩大或偷换适用条件。无条件错误填 true。
5. 失败、取消、缺失记录和无发布正文仍须审阅；不得用草稿补答、删题或按正确拒答计分。
   失败项覆盖必须为 []，action_correct 为 false。在 notes 写明判据和引用原文位置。
6. 只有本人完成核对，才填写 reviewer、带时区的 reviewed_at（ISO 8601，例如
   2026-11-15T18:00:00+08:00），并将该答案 independent_human_review 改为 true。
   未审项保留 false/null。不要让 Agent 代填后声明是真人，不要复制模型 Verifier 的判断。
7. 不要修改题目/答案、blind_id、question_id、label、引用 ID、relation 或包标识。
   将填写后的 scorecard.json 交回操作者；可先交部分，未完成的整组质量率保持 null。

阅读册保留了发布正文的条件、引用、缺口和澄清原话；仅隐藏配置、模型、运行与自评元数据。
原话和文风仍可能透露流程特征，因此这是匿名评审材料，不保证评审者完全无法推测来源。
本包没有原 Agent 分数、系统映射或自动填写的真人判分。不同真人的分歧应分别留原表，
由实际评审者核对原文裁决，禁止自动用多数票或 Agent 意见冒充独立复核。
"""


def export_packet(root, corpus_root, output):
    root, corpus_root, output = root.resolve(), corpus_root.resolve(), output.resolve()
    manifest = read_json(root / "manifest.json")
    template = make_template(root)
    files = corpus_files(corpus_root, manifest)
    modes = modes_for(manifest)
    if len(modes) > 26:
        raise ValueError("匿名包最多支持 26 个配置")
    by_identity = {(r["mode"], r["question_id"]): r for r in template["reviews"]}
    # Read before creating the destination; an invalid source must not produce a usable packet.
    rows = {}
    for mode in modes:
        for path in sorted((root / mode).glob("*.json")):
            row = read_json(path)
            rows[(mode, row["question_id"])] = row
    packet_id = uuid.uuid4().hex
    scorecard = {"version": 1, "review_source": "human", "packet_id": packet_id,
                 "booklet_sha256": None, "question_reviews": [], "reviews": []}
    bindings, book = [], ["# 匿名问答试评阅读册\n", "请先阅读 README.md；评分草案仍待真人核对。\n"]
    for question in manifest["questions"]:
        qid = question["question_id"]
        rubric = {k: question.get(k) for k in ("question", "category", "expected_action", "required_claims",
                                              "forbidden_conclusions", "absence_review_scope")}
        book.extend([f"## {qid}\n", "### 题目与待审核评分草案\n", block(rubric)])
        scorecard["question_reviews"].append({"question_id": qid, "rubric_correct": None,
                                              "reviewer": "", "reviewed_at": "", "notes": ""})
        ordered_modes = list(modes)
        random.SystemRandom().shuffle(ordered_modes)
        for index, mode in enumerate(ordered_modes):
            label = chr(ord("A") + index)
            blind_id = f"{qid}_{label}"
            shown, claim_map, evidence_map = blind_answer(rows.get((mode, qid), {}))
            book.extend([f"### 答案 {label}（{blind_id}）\n", block(shown)])
            source_review = by_identity[(mode, qid)]
            links = [{"claim_id": claim_map[link["claim_id"]],
                      "evidence_id": evidence_map[link["evidence_id"]], "relation": link["relation"],
                      "supports_claim": None, "relation_correct": None} for link in source_review["links"]]
            scorecard["reviews"].append({"blind_id": blind_id, "question_id": qid, "label": label,
                                          "reviewer": "", "reviewed_at": "", "independent_human_review": False,
                                          "links": links, "covered_required_keys": None,
                                          "forbidden_conclusions_present": None, "action_correct": None,
                                          "conditions_correct": None, "notes": ""})
            bindings.append({"blind_id": blind_id, "mode": mode, "question_id": qid,
                             "result_sha256": source_review["result_sha256"],
                             "claim_map": claim_map, "evidence_map": evidence_map})
    booklet = "\n".join(book)
    scorecard["booklet_sha256"] = hashlib.sha256(booklet.encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    reviewer = output / "reviewer"
    reviewer.mkdir()
    for name, path in files.items():
        destination = reviewer / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    instructions = INSTRUCTIONS.replace("本包用于现有结果的试评；",
                                        f"本次共 {len(manifest['questions'])} 题、{len(bindings)} 份答案记录。")
    (reviewer / "README.md").write_text(instructions, encoding="utf-8", newline="\n")
    (reviewer / "review-book.md").write_text(booklet, encoding="utf-8", newline="\n")
    (reviewer / "scorecard.json").write_text(json_text(scorecard), encoding="utf-8", newline="\n")
    packet_files = {p.relative_to(reviewer).as_posix(): sha256(p) for p in sorted(reviewer.rglob("*"))
                    if p.is_file() and p.name != "scorecard.json"}
    key = {"version": 1, "run_directory": str(root), "corpus_directory": str(corpus_root),
           "manifest_sha256": template["manifest_sha256"], "template": scorecard, "bindings": bindings,
           "source_files": {name: sha256(path) for name, path in files.items()}, "packet_files": packet_files}
    (output / "operator-key.json").write_text(json_text(key), encoding="utf-8", newline="\n")
    with zipfile.ZipFile(output / "reviewer.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(reviewer.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(reviewer).as_posix())
    return output


def signed_review(entry):
    if not isinstance(entry.get("reviewer"), str) or not entry["reviewer"].strip():
        raise ValueError("已审条目必须填写真人审核者标识")
    try:
        timestamp = datetime.fromisoformat(entry["reviewed_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("已审条目必须填写带时区的 ISO 8601 审核时间") from exc
    if timestamp.utcoffset() is None:
        raise ValueError("审核时间必须包含时区")
    if not isinstance(entry.get("notes"), str) or not entry["notes"].strip():
        raise ValueError("已审条目必须记录原文依据和判分理由")


def index_entries(entries, field):
    indexed = {}
    for entry in entries:
        identity = entry[field]
        if identity in indexed:
            raise ValueError("评分表包含重复条目")
        indexed[identity] = entry
    return indexed


def link_identity(link):
    return link["claim_id"], link["evidence_id"], link["relation"]


def import_scorecard(root, key_path, scorecard_path):
    """Return a validated annotation and separate summary; never mutate frozen records."""
    root, key_path = root.resolve(), key_path.resolve()
    key, filled = read_json(key_path), read_json(scorecard_path)
    expected = key["template"]
    template = make_template(root)
    if key.get("version") != 1 or key["manifest_sha256"] != template["manifest_sha256"]:
        raise ValueError("源结果 manifest SHA 已变化，须重新评审")
    for name, digest in key["packet_files"].items():
        if sha256(local_file(key_path.parent / "reviewer", name)) != digest:
            raise ValueError("匿名评审材料 SHA 已变化，须重新导出")
    corpus_root = Path(key["corpus_directory"])
    corpus_files(corpus_root, read_json(root / "manifest.json"))
    if any(sha256(local_file(corpus_root, name)) != digest for name, digest in key["source_files"].items()):
        raise ValueError("冻结来源 SHA 已变化，须重新评审")
    if any(filled.get(k) != expected[k] for k in ("version", "review_source", "packet_id", "booklet_sha256")):
        raise ValueError("评分表与匿名包身份不匹配；只接受本包的真人评分表")
    reviews = index_entries(filled["reviews"], "blind_id")
    rubrics = index_entries(filled["question_reviews"], "question_id")
    if (set(reviews) != {r["blind_id"] for r in expected["reviews"]}
            or set(rubrics) != {q["question_id"] for q in expected["question_reviews"]}):
        raise ValueError("不得增加或删除答案/题目条目；未审条目保持待审")
    bindings = index_entries(key["bindings"], "blind_id")
    original = {(r["mode"], r["question_id"]): r for r in template["reviews"]}
    questions = {q["question_id"]: q for q in read_json(root / "manifest.json")["questions"]}
    records = {mode: load_records(root / mode, questions) for mode in modes_for(read_json(root / "manifest.json"))}
    if ({(b["mode"], b["question_id"]) for b in bindings.values()} != set(original)
            or len(bindings) != len(original)):
        raise ValueError("操作者映射缺少或重复源结果")
    for item in expected["reviews"]:
        decision, binding = reviews[item["blind_id"]], bindings[item["blind_id"]]
        target = original[(binding["mode"], binding["question_id"])]
        if binding["result_sha256"] != target["result_sha256"]:
            raise ValueError("源逐题结果 SHA 已变化，须重新评审")
        if any(decision.get(k) != item[k] for k in ("question_id", "label")):
            raise ValueError("匿名题号或标签被修改")
        expected_links = {link_identity(link) for link in item["links"]}
        actual_links = [link_identity(link) for link in decision["links"]]
        if set(actual_links) != expected_links or len(actual_links) != len(expected_links):
            raise ValueError("匿名引用关系被增加、删除或修改")
        if type(decision.get("independent_human_review")) is not bool:
            raise ValueError("真人审核声明必须是布尔值")
        if decision["independent_human_review"] is not True:
            continue
        rubric = rubrics[decision["question_id"]]
        if rubric.get("rubric_correct") is not True:
            raise ValueError("本题评分草案未获真人确认或存在异议；先裁决，不能直接计分")
        signed_review(rubric)
        signed_review(decision)
        if any(type(decision.get(k)) is not bool for k in ("action_correct", "conditions_correct")):
            raise ValueError("真人必须判断动作与适用条件")
        row = records[binding["mode"]].get(binding["question_id"], ({}, None))[0]
        row_completed = not row.get("technical_error") and row.get("run", {}).get("status") == "completed"
        # The existing scorer validates failed/missing coverage; no failed task is a correct action.
        if not row_completed and decision["action_correct"]:
            raise ValueError("失败/取消/缺失题不能标为动作正确")
        claim_ids = {v: k for k, v in binding["claim_map"].items()}
        evidence_ids = {v: k for k, v in binding["evidence_map"].items()}
        target.update({k: copy.deepcopy(decision[k]) for k in ("reviewer", "reviewed_at",
                      "independent_human_review", "covered_required_keys", "forbidden_conclusions_present",
                      "action_correct", "conditions_correct", "notes")})
        target["links"] = [{**link, "claim_id": claim_ids[link["claim_id"]],
                            "evidence_id": evidence_ids[link["evidence_id"]]} for link in decision["links"]]
    template.update(packet_id=expected["packet_id"], question_reviews=copy.deepcopy(filled["question_reviews"]))
    manifest = read_json(root / "manifest.json")
    modes = modes_for(manifest)
    metrics = {mode: summarize_manual(root, mode, template) for mode in modes}
    return template, {"note": "真人声明由填写者提供；程序不认证身份。原始结果不变，未完整审核不出总体质量率。",
                      "packet_id": expected["packet_id"], "manual_metrics": metrics,
                      "objective_counts": {mode: summarize(root / mode) for mode in modes},
                      "supplemental_note": "逐题动作/条件判断保存在 human-reviews.json，现有人审汇总不统计这两项。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("run_directory", type=Path)
    export.add_argument("--corpus", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("run_directory", type=Path)
    ingest.add_argument("--key", type=Path, required=True)
    ingest.add_argument("--scorecard", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        print(export_packet(args.run_directory, args.corpus, args.output))
    else:
        annotation, report = import_scorecard(args.run_directory, args.key, args.scorecard)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "human-reviews.json").write_text(json_text(annotation), encoding="utf-8")
        (args.output / "summary.json").write_text(json_text(report), encoding="utf-8")
        print(args.output.resolve())


if __name__ == "__main__":
    main()
