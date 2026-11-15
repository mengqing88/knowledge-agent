from sqlalchemy import select

from app.core.access import check_snapshot_access, corpus_role
from app.core.errors import DomainError
from app.core.models import (
    EvidenceSpan,
    MCPInvocation,
    Paper,
    PaperVersion,
    ParseRevision,
    QuestionRun,
    Snapshot,
    SourceBlock,
    uid,
)


def allowed_parse_ids(snapshot: Snapshot) -> set[str]:
    return {item["parse_revision_id"] for item in snapshot.manifest}


def validate_range(text: str, start: int, end: int):
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise DomainError("引用字符范围需要为有效整数且不能越界")


def validate_span(db, snapshot: Snapshot, block_id: str, start: int, end: int, quote: str) -> SourceBlock:
    block = db.get(SourceBlock, block_id)
    if not block or block.parse_revision_id not in allowed_parse_ids(snapshot):
        raise DomainError("引用不属于当前语料快照", 403)
    validate_range(block.text, start, end)
    if block.text[start:end] != quote:
        raise DomainError("引用摘录与保存的原文不一致")
    return block


def get_or_create_span(db, run_id: str, snapshot: Snapshot, block_id: str, start: int, end: int):
    block = db.get(SourceBlock, block_id)
    if not block:
        raise DomainError("来源块不存在", 404)
    validate_range(block.text, start, end)
    quote = block.text[start:end]
    validate_span(db, snapshot, block_id, start, end, quote)
    existing = db.scalar(select(EvidenceSpan).where(
        EvidenceSpan.run_id == run_id, EvidenceSpan.block_id == block_id,
        EvidenceSpan.char_start == start, EvidenceSpan.char_end == end,
    ))
    if existing:
        return existing
    span = EvidenceSpan(id=uid(), run_id=run_id, block_id=block_id, char_start=start, char_end=end, quote=quote)
    db.add(span)
    db.flush()
    return span


def evidence_data(db, span: EvidenceSpan, owner_id: str) -> dict:
    run = db.get(QuestionRun, span.run_id)
    snapshot = db.get(Snapshot, run.snapshot_id) if run else None
    if not snapshot or not run:
        raise DomainError("证据不可访问", 404)
    check_snapshot_access(db, snapshot, owner_id)
    role = corpus_role(db, snapshot.corpus_id, owner_id)
    is_owner = run.owner_id == owner_id
    owner_execution = is_owner and run.status in {"queued", "running"}
    reviewer_pending = run.status == "review_pending" and role in {"admin", "editor"}
    if run.status != "completed" and not (owner_execution or reviewer_pending):
        raise DomainError("答案尚未通过审核", 404)
    if span.invocation_id:
        row = validate_evidence(db, snapshot, span)
        if row.server_id == "project_runtime_logs":
            if not is_owner and role not in {"admin", "editor"}:
                raise DomainError("私人运行日志仅对本人和审核角色开放", 404)
        else:
            from app.core.config import get_settings
            from app.integrations.mcp_client import configured_server
            configured_server(get_settings(), row.server_id, owner_id, snapshot.corpus_id)
        return {"id": span.id, "quote": span.quote, "block_id": row.id,
                "char_start": span.char_start, "char_end": span.char_end,
                "title": f"工具观察：{row.server_id}/{row.tool_name}", "paper_id": "",
                "paper_version_id": row.id, "parse_revision_id": "", "sha256": span.source_sha256,
                "pdf_page_index": None, "pdf_page_number": None, "bbox": [], "printed_page_label": None,
                "section": row.tool_name, "location_kind": "mcp_observation",
                "location_label": f"工具记录 · {row.created_at.isoformat()}",
                "file_url": f"/api/evidence/{span.id}", "source_kind": "mcp_observation",
                "invocation_id": row.id, "recorded_at": row.created_at.isoformat(),
                "knowledge_status": "external_observation_not_indexed"}
    block = validate_span(db, snapshot, span.block_id, span.char_start, span.char_end, span.quote)
    parse = db.get(ParseRevision, block.parse_revision_id)
    version = db.get(PaperVersion, parse.paper_version_id)
    paper = db.get(Paper, version.paper_id)
    is_pdf = version.filename.lower().endswith(".pdf")
    return {"id": span.id, "quote": span.quote, "block_id": block.id,
            "char_start": span.char_start, "char_end": span.char_end,
            "title": next(m["title"] for m in snapshot.manifest if m["paper_version_id"] == version.id),
            "paper_id": paper.id, "paper_version_id": version.id,
            "parse_revision_id": parse.id, "sha256": version.sha256,
            "pdf_page_index": block.page_index, "pdf_page_number": block.page_index + 1,
            "printed_page_label": block.printed_page_label, "bbox": block.bbox,
            "section": block.section, "location_kind": "pdf_page" if is_pdf else "section",
            "location_label": f"PDF 第 {block.page_index + 1} 页" if is_pdf else block.section or "正文",
            "file_url": f"/api/paper-versions/{version.id}/file" +
                        (f"#page={block.page_index + 1}" if is_pdf else "")}


def observation_text(result):
    import json
    texts = [item["text"] for item in result.get("content", [])
             if item.get("type") == "text" and isinstance(item.get("text"), str)]
    return "\n".join(texts) if texts else json.dumps(result.get("structuredContent", {}), ensure_ascii=False, sort_keys=True)


def validate_evidence(db, snapshot, span):
    if not span.invocation_id:
        if not span.block_id:
            raise DomainError("证据没有有效来源", 403)
        return validate_span(db, snapshot, span.block_id, span.char_start, span.char_end, span.quote)
    from app.agents.governance import digest
    run = db.get(QuestionRun, span.run_id)
    row = db.get(MCPInvocation, span.invocation_id)
    if (span.block_id or not run or run.snapshot_id != snapshot.id or not row or
            row.run_id != run.id or row.owner_id != run.owner_id or row.status != "succeeded" or
            digest(row.result) != span.source_sha256):
        raise DomainError("外部证据身份或回执校验失败", 403)
    text = observation_text(row.result)
    validate_range(text, span.char_start, span.char_end)
    if text[span.char_start:span.char_end] != span.quote:
        raise DomainError("外部证据摘录与回执不一致", 403)
    return row
