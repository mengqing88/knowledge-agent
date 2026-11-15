"""Freeze licensed official excerpts and an Agent-authored 12-question rubric.

Sources are vendored from immutable release tags. This script never calls a model.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "evaluation/public_docs_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = json.loads((TARGET / "sources.json").read_text(encoding="utf-8"))
    for entry in sources:
        if sha(TARGET / "originals" / entry["file"]) != entry["sha256"]:
            raise ValueError("Original source changed")
    specs = [
        ("pg-levels", "PostgreSQL 17.6 isolation levels", "postgres-mvcc.sgml", 269, 285),
        ("pg-read-committed", "PostgreSQL 17.6 Read Committed", "postgres-mvcc.sgml", 317, 333),
        ("pg-repeatable-read", "PostgreSQL 17.6 Repeatable Read", "postgres-mvcc.sgml", 492, 562),
        ("celery-acks", "Celery 5.5.3 acknowledgement", "celery-tasks.rst", 16, 47),
        ("celery-retry", "Celery 5.5.3 retry semantics", "celery-tasks.rst", 579, 614),
        ("celery-delay", "Celery 5.5.3 retry delay", "celery-tasks.rst", 646, 663),
    ]
    documents = []
    for key, title, original, first, last in specs:
        lines = (TARGET / "originals" / original).read_text(encoding="utf-8").splitlines()
        text = "\n".join(lines[first - 1:last])
        if original.endswith("sgml"):
            text = re.sub(r"<[^>]+>", "", text)
            text = "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n") if p.strip())
        else:
            text = "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n") if p.strip())
        path = TARGET / f"{key}.txt"
        path.write_text(text + "\n", encoding="utf-8")
        source = next(s for s in sources if s["file"] == original)
        documents.append(dict(id=key, title=title, filename=path.name, sha256=sha(path),
                              source=source, original_lines=[first, last],
                              transformation="strip SGML inline tags; normalize whitespace; no paraphrase",
                              license="PostgreSQL License" if key.startswith("pg-") else "BSD-3-Clause",
                              license_file="originals/postgres-COPYRIGHT" if key.startswith("pg-") else "originals/celery-LICENSE"))

    def point(key, meaning, document, quote):
        text = (TARGET / f"{document}.txt").read_text(encoding="utf-8")
        if quote not in text:
            raise ValueError(f"Nonverbatim evidence: {key}")
        return dict(key=key, meaning=meaning, evidence=[dict(document=document, quote=quote)])

    default = point("default", "默认隔离级别为 Read Committed", "pg-read-committed",
                    "Read Committed is the default isolation level in PostgreSQL.")
    uncommitted = point("mapping", "Read Uncommitted 实际行为同 Read Committed", "pg-levels",
                        "PostgreSQL's Read Uncommitted mode behaves like Read Committed.")
    phantom = point("phantom", "PostgreSQL 的 Repeatable Read 不允许幻读", "pg-levels",
                    "PostgreSQL's Repeatable Read implementation does not allow phantom reads.")
    idem = point("idempotency", "启用延迟确认的前提是任务幂等", "celery-acks",
                 "If your task is idempotent you can set the :attr:`~Task.acks_late` option to have the worker acknowledge the message *after* the task returns instead.")
    loss = point("loss", "执行子进程终止时即使启用延迟确认仍会确认；这些情形需考虑 task_reject_on_worker_lost", "celery-acks",
                 "even when :attr:`~Task.acks_late` is enabled.")
    retry = point("retry_identity", "retry 发送新消息但保留 task-id 并投递原队列", "celery-retry",
                  "When you call ``retry`` it'll send a new message, using the same task-id, and it'll take care to make sure the message is delivered to the same queue as the originating task.")
    exception = point("retry_control", "retry 通常抛出 Retry，后续代码不执行；throw=False 为例外", "celery-retry",
                      "The :meth:`Task.retry() <@Task.retry>` call will raise an exception so any code after the retry won't be reached.")
    rc = point("rc_snapshot", "RC 使用每条查询开始时快照；其他事务中间提交可使两次 SELECT 不同", "pg-read-committed",
               "two successive SELECT commands can see different data, even though they are within a single transaction")
    rr = point("rr_snapshot", "RR 从事务首条非事务控制语句取快照，连续 SELECT 不见其他事务后提交", "pg-repeatable-read",
               "repeatable read transaction sees a snapshot as of the start of the first non-transaction-control statement in the transaction")
    restart = point("whole_retry", "遇该并发更新序列化失败应中止并从头重试整个事务", "pg-repeatable-read",
                    "it should abort the current transaction and retry the whole transaction from the beginning.")
    delay = point("delay", "默认重试间隔三分钟，单位秒，countdown 可覆盖", "celery-delay",
                  "By default this is set to 3 minutes. Note that the unit for setting the delay is in seconds (int or float).")
    questions = []

    def q(key, category, text, action, points, forbidden, absence=None):
        questions.append(dict(question_id=key, category=category, question=text, expected_action=action,
                              required_claims=points, forbidden_conclusions=forbidden, absence_review_scope=absence,
                              agent_reviewed=True, human_reviewed=False, fictional=False,
                              rubric_review="Codex authored and checked against frozen original excerpts; not independent human review"))

    q("q01", "fact", "PostgreSQL 文档中的默认事务隔离级别是什么？", "supported", [default], ["默认是 Serializable"])
    q("q02", "fact", "PostgreSQL 请求 Read Uncommitted 时，实际行为等同哪个级别？", "supported", [uncommitted], ["允许读取其他事务未提交的数据"])
    q("q03", "fact", "Celery 调用 Task.retry() 后，task-id 和投递队列会怎样？", "supported", [retry], ["每次必然生成新 task-id"])
    q("q04", "fact", "PostgreSQL 的 Repeatable Read 是否允许幻读？请限定到该数据库实现。", "supported", [phantom], ["所有数据库的 Repeatable Read 都禁止幻读"])
    q("q05", "compound", "比较 PostgreSQL Read Committed 和 Repeatable Read 的快照时点，以及连续两次 SELECT 能否看到其他事务新提交的变化。", "supported", [rc, rr], ["两种级别都固定每次 SELECT 的相同快照"])
    q("q06", "compound", "Celery 的 acks_late 适合什么任务？启用后执行子进程终止是否仍可能确认消息，若希望这种情形重投应考虑什么设置？", "supported", [idem, loss], ["acks_late 单独保证任何崩溃都重投", "恰好执行一次"])
    q("q07", "compound", "PostgreSQL Repeatable Read 遇到并发更新产生序列化失败该重试什么？Celery retry() 是否正常返回并执行后续代码，有什么例外？", "supported", [restart, exception], ["只重试失败的 SQL 即可", "retry 默认正常返回"])
    q("q08", "partial", "Celery 默认重试间隔和计时单位是什么，怎样覆盖？据此能保证我们线上 API 的 P99 小于 100 ms 吗？", "partial", [delay], ["保证线上 P99 小于 100 ms"], "六篇冻结摘录不包含线上负载、延迟和 SLA 数据")
    q("q09", "partial", "PostgreSQL 默认隔离级别是什么？它在我这台机器上处理百万行更新要多少毫秒？", "partial", [default], ["编造本机百万行更新耗时"], "六篇冻结摘录没有用户机器性能实验")
    q("q10", "insufficient", "仅凭当前材料，给出我们团队生产 Celery 队列昨晚最大积压数和当时根因。", "insufficient", [], ["编造昨晚队列指标或根因"], "六篇冻结摘录是公开技术说明，不含团队生产日志")
    q("q11", "insufficient", "材料有没有 PostgreSQL 17.6 与 MySQL 8.4 在同一硬件上的吞吐对照？请给出倍数。", "insufficient", [], ["编造吞吐或相对倍数"], "六篇冻结摘录没有 MySQL 对照或实测吞吐")
    q("q12", "clarification", "把它调高一点能解决这个问题吗？", "needs_clarification", [], ["假定具体参数或故障并提供确定结论"], "缺少参数名、系统、故障现象和目标；应先澄清")
    corpus = dict(version=1, scope="Six selected official excerpts only; not complete manuals", documents=documents)
    path = TARGET / "corpus.json"
    path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    suite = dict(version=1, review_source="agent", corpus_manifest="evaluation/public_docs_v1/corpus.json",
                 corpus_manifest_sha256=sha(path), questions=questions,
                 modes=["baseline_rag", "full_agent"], repeats=1,
                 limitations="Agent-authored and Agent-scored small sample; not independent human or business accuracy")
    (TARGET / "suite.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared", len(documents), "documents and", len(questions), "Agent-reviewed questions")


if __name__ == "__main__":
    main()
