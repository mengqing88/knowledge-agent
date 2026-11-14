"""Generate fictional long documents and exhaustive per-question source spans."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "chunking"

TOPICS = [
    ("access", "权限申请", [
        ("临时外包人员需要生产只读权限时，应由谁批准，最多有效多久？",
         "外包人员的生产只读权限必须同时经过业务负责人和安全管理员批准，有效期最多为七十二小时。"),
        ("员工离职后账号需要在什么时间内关闭？",
         "员工离职交接完成后，身份管理员应在三十分钟内停用登录账号并撤销仍有效的访问令牌。"),
        ("紧急访问的审计记录保存多久？",
         "紧急访问必须记录工单编号、操作人和资源范围，审计记录至少保留一百八十天。"),
    ]),
    ("billing", "费用报销", [
        ("缺少发票时能否仅凭付款截图报销？",
         "付款截图不能替代正式发票，缺失发票的申请应退回补充材料，不得由系统自动批准。"),
        ("跨境差旅的汇率按哪一天计算？",
         "跨境差旅报销采用实际支付日的银行结算汇率，而非提交申请当天的公开参考汇率。"),
        ("重复付款发现后，财务需要何时通知申请人？",
         "发现重复付款后，财务应在两个工作日内通知申请人，并登记退款进展。"),
    ]),
    ("operations", "运行维护", [
        ("关键服务发生一级事故后，多久应启动应急响应？",
         "关键服务发生一级事故后，值班负责人应在五分钟内启动响应，并指定唯一事故协调人。"),
        ("恢复服务后是否可以立即关闭事故工单？",
         "服务恢复不代表事故工单可以立即关闭；应先完成数据一致性核验，并由事故协调人确认。"),
        ("备份恢复演练失败时如何处理？",
         "备份恢复演练失败必须暂停相关数据迁移，修复后重新演练，不能用备份任务成功代替恢复验证。"),
    ]),
    ("release", "发布管理", [
        ("灰度发布出现错误率上升时怎么办？",
         "灰度发布期间，如果错误率连续三个采样窗口超过基线两倍，应停止扩量并回滚到上一稳定版本。"),
        ("数据库变更发布前有哪些审核要求？",
         "涉及数据库结构的发布必须提供兼容性说明和恢复方案，并由数据库负责人完成审核。"),
        ("变更发布后需要观察多长时间？",
         "常规变更发布后至少观察四十分钟，观察结束前不允许删除上一稳定版本的部署产物。"),
    ]),
]


def main():
    OUT.mkdir(exist_ok=True)
    documents, questions = [], []
    for doc_id, title, facts in TOPICS:
        paragraphs = []
        # Three long paragraphs per file exceed the largest token window. Facts
        # occur at different positions among plausible but non-answering text.
        for number, (query, quote) in enumerate(facts):
            sentences = [
                f"本节是虚构的{title}培训材料第{number + 1}单元，案例编号为{doc_id}-{i:02d}。"
                "处理记录应描述背景、范围和当前进展，材料由对应负责人维护，具体操作应与工单关联。"
                for i in range(26)
            ]
            sentences.insert(7 + number * 6, quote)
            paragraphs.append("".join(sentences))
            questions.append({"id": f"{doc_id}-{number}", "question": query,
                              "split": "held_out" if number == 2 else "development",
                              "evidence": [{"document": doc_id, "quote": quote}]})
        text = f"# 虚构{title}手册\n\n" + "\n\n".join(paragraphs) + "\n"
        filename = f"{doc_id}.md"
        (OUT / filename).write_text(text, encoding="utf-8", newline="\n")
        documents.append({"id": doc_id, "filename": filename,
                          "sha256": hashlib.sha256(text.encode()).hexdigest()})
    structured = {
        "roles.csv": "name,role,approval\nLin,reader,security-admin\nMei,editor,owner\n",
        "service.json": '{"service":{"timeout_seconds":45,"retry_limit":2},"audit":{"retention_days":180}}\n',
    }
    for filename, content in structured.items():
        (OUT / filename).write_text(content, encoding="utf-8", newline="\n")
        documents.append({"id": filename, "filename": filename,
                          "sha256": hashlib.sha256(content.encode()).hexdigest()})
    questions.extend([
        {"id": "roles", "split": "development", "question": "Mei 的角色和审批人分别是什么？",
         "evidence": [{"document": "roles.csv", "quote": "name: Mei | role: editor | approval: owner"}]},
        {"id": "timeout", "split": "held_out", "question": "service.timeout_seconds 配置值是多少？",
         "evidence": [{"document": "service.json", "section": "service.timeout_seconds", "quote": "45"}]},
        {"id": "joint", "split": "held_out", "question": "上线与故障处理分别在何种情况下需要回滚或核验？",
         "evidence": [{"document": "release", "quote": TOPICS[3][2][0][1]},
                      {"document": "operations", "quote": TOPICS[2][2][1][1]}]},
    ])
    manifest = {"version": 1, "fictional": True, "human_reviewed": False,
                "purpose": "Long-document chunk retrieval control experiment; not business quality acceptance",
                "documents": documents, "questions": questions}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(documents)} fictional documents and {len(questions)} labelled questions")


if __name__ == "__main__":
    main()
