import hashlib
from zipfile import ZIP_DEFLATED, ZipFile

import pymupdf
import pytest
from app.core.errors import DomainError
from app.core.models import Chunk, PaperVersion, SourceBlock
from app.knowledge import ingestion
from app.knowledge.ingestion import build_chunks, extract_document, extract_pdf, ingest_version
from sqlalchemy import func, select


def test_pdf_parser_page_positions_and_empty_scan(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for number in range(2):
        page = doc.new_page()
        page.insert_text((72, 100), f"Fictional test page {number+1}. Method M F1 82 on dataset X.")
    doc.save(path)
    doc.close()
    blocks, pages, warnings = extract_pdf(path)
    assert pages == 2
    assert {b["page_index"] for b in blocks} == {0, 1}
    assert blocks[0]["bbox"][0] == 72
    assert warnings
    doc = pymupdf.open()
    doc.new_page()
    blank = tmp_path / "blank.pdf"
    doc.save(blank)
    doc.close()
    with pytest.raises(DomainError, match="OCR"):
        extract_pdf(blank)
    invalid = tmp_path / "invalid.pdf"
    invalid.write_bytes(b"not PDF")
    with pytest.raises(DomainError, match="损坏"):
        extract_pdf(invalid)


def test_pdf_ruled_tables_keep_headers_rows_and_page_geometry(tmp_path):
    path = tmp_path / "ruled-table.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=400)
    page.insert_text((50, 48), "Results summary before the table.")
    xs, ys = [50, 180, 280, 380], [80, 120, 160, 200]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    rows = [["Metric", "Method M", "Baseline"], ["F1", "82", "80"],
            ["Latency", "120 ms", "150 ms"]]
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            page.insert_text((xs[column_index] + 8, ys[row_index] + 25), value, fontsize=11)
    page.insert_text((50, 245), "Results summary after the table.")
    doc.save(path)
    doc.close()

    blocks, pages, warnings = extract_pdf(path)
    table_blocks = [block for block in blocks if block.get("section", "").startswith("PDF 第 1 页 · 表格 1")]
    assert pages == 1
    assert len(table_blocks) == 2
    assert "Metric: F1 | Method M: 82 | Baseline: 80" == table_blocks[0]["text"]
    assert "Metric: Latency | Method M: 120 ms | Baseline: 150 ms" == table_blocks[1]["text"]
    assert table_blocks[0]["section"].endswith("第 2 行")
    assert table_blocks[0]["bbox"][1] < table_blocks[1]["bbox"][1]
    assert any("before the table" in block["text"] for block in blocks)
    assert any("after the table" in block["text"] for block in blocks)
    assert not any(block["text"].strip() == "F1 82 80" for block in blocks)
    assert any("有边框表格按表头和行结构提取" in warning for warning in warnings)


def test_chunks_keep_raw_source_offsets_and_page_boundaries():
    a = SourceBlock(id="a", page_index=0, block_index=0, text="A negation is not re-\nmoved.\n",
                    bbox=[72, 100, 500, 140], parse_revision_id="parse")
    b = SourceBlock(id="b", page_index=1, block_index=0, text="120 ms != 100 ms.",
                    bbox=[72, 100, 500, 140], parse_revision_id="parse")
    chunks = build_chunks([a, b], {0: 842, 1: 842})
    assert len(chunks) == 2
    assert "not removed" in chunks[0]["text"]
    assert chunks[0]["source_ranges"] == [{"block_id": "a", "char_start": 0, "char_end": len(a.text)}]
    assert a.text.endswith("moved.\n")


def test_repeated_headers_do_not_remove_matching_body_evidence():
    text = "Escalate to the incident commander."
    headers = [SourceBlock(id=f"header-{page}", page_index=page, block_index=0,
                           text=text, bbox=[72, 10, 500, 30], parse_revision_id="parse")
               for page in range(3)]
    body = SourceBlock(id="body", page_index=2, block_index=1, text=text,
                       bbox=[72, 200, 500, 240], parse_revision_id="parse")
    chunks = build_chunks([*headers, body], {page: 842 for page in range(3)})
    assert [r["block_id"] for c in chunks for r in c["source_ranges"]] == ["body"]
    assert chunks[0]["text"] == text
    assert body.text == text


def test_repeated_margin_blocks_on_one_page_are_not_cross_page_headers():
    blocks = [SourceBlock(id=f"block-{i}", page_index=0, block_index=i,
                          text="Important warning.", bbox=[72, 10, 500, 30], parse_revision_id="parse")
              for i in range(3)]
    chunks = build_chunks(blocks, {0: 842})
    assert {r["block_id"] for c in chunks for r in c["source_ranges"]} == {b.id for b in blocks}


def test_text_and_structured_formats_preserve_searchable_source_locations(tmp_path):
    fixtures = {
        "guide.txt": "# 权限申请\n\n新人通过管理员申请只读权限。\n",
        "team.csv": "name,role\nLin,editor\nMei,reader\n",
        "team.tsv": "name\trole\nLin\teditor\n",
        "settings.json": '{"service":{"timeout":45},"enabled":true}',
        "events.jsonl": '{"id":1,"message":"started"}\n{"id":2,"message":"finished"}\n',
        "guide.html": (
            "<html><h1>发布流程</h1><p>发布前必须由负责人复核。</p>"
            "<script>secret should not be indexed</script><div hidden>hidden secret</div>"
            "<template>template secret</template>"
            "<p style='display: none'>also hidden</p></html>"
        ),
    }
    for filename, content in fixtures.items():
        path = tmp_path / filename
        path.write_text(content, encoding="utf-8")
        blocks, pages, warnings = extract_document(path, filename)
        text = "\n".join(block["text"] for block in blocks)
        assert pages is None and warnings
        assert text
        assert all(block["section"] for block in blocks)
        if filename.endswith(".csv") or filename.endswith(".tsv"):
            assert "role: editor" in text and any("第 2 行" in block["section"] for block in blocks)
        if filename == "settings.json":
            assert any(block["section"] == "service.timeout" and block["text"] == "45" for block in blocks)
        if filename == "events.jsonl":
            assert any("第 2 行" in block["section"] and "finished" in block["text"] for block in blocks)
        if filename.endswith(".html"):
            assert "负责人复核" in text and "secret" not in text


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "gb18030"])
def test_txt_parser_accepts_common_chinese_encodings(tmp_path, encoding):
    expected = "七堇年作品中的原文检索证据。"
    path = tmp_path / "novel.txt"
    path.write_bytes(expected.encode(encoding))

    blocks, pages, warnings = extract_document(path, path.name)

    assert pages is None and warnings
    assert "".join(block["text"] for block in blocks) == expected


def test_txt_parser_still_rejects_binary_data(tmp_path):
    path = tmp_path / "novel.txt"
    path.write_bytes(b"\x00\xff\x10")

    with pytest.raises(DomainError):
        extract_document(path, path.name)


def test_xlsx_parser_keeps_sheet_and_row_location(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "成员"
    sheet.append(["姓名", "角色"])
    sheet.append(["Lin", "编辑者"])
    sheet.append(["统计", "=COUNTA(A2:A2)"])
    workbook.create_sheet("空表")
    hidden = workbook.create_sheet("隐藏数据")
    hidden.sheet_state = "hidden"
    hidden.append(["属性", "值"])
    hidden.append(["审批人", "管理员"])
    path = tmp_path / "team.xlsx"
    workbook.save(path)
    workbook.close()

    blocks, pages, warnings = extract_document(path, path.name)
    assert pages is None
    assert any("工作表：成员" in block["section"] and "角色: 编辑者" in block["text"] for block in blocks)
    assert any("工作表：成员 · 第 3 行" == block["section"] and "=COUNTA(A2:A2)" in block["text"]
               for block in blocks)
    assert any("工作表：隐藏数据" in block["section"] and "属性: 审批人" in block["text"]
               and "值: 管理员" in block["text"]
               for block in blocks)
    assert any("公式文本索引" in warning for warning in warnings)


def test_xlsx_parser_rejects_corrupt_archives_and_extreme_compression(tmp_path):
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"PK\x03\x04placeholder")
    with pytest.raises(DomainError, match="XLSX"):
        extract_document(corrupt, corrupt.name)

    compressed = tmp_path / "compressed.xlsx"
    with ZipFile(compressed, "w", ZIP_DEFLATED) as archive:
        archive.writestr("payload", b"a" * (1024 * 1024 + 1))
    with pytest.raises(DomainError, match="压缩比例"):
        extract_document(compressed, compressed.name)


@pytest.mark.parametrize("filename,content", [
    ("bad.json", "{not-json}"),
    ("binary.csv", "a,b\n\x00,1"),
    ("broken.tsv", "a\tb\n\"unterminated\t1"),
])
def test_structured_parser_rejects_invalid_inputs(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DomainError):
        extract_document(path, filename)


def test_ingestion_retry_no_duplicate_chunks(sessions, settings, corpus, fake):
    settings.storage_dir.mkdir()
    path = settings.storage_dir / "a.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Fictional teaching paper. M achieved F1 82 compared with baseline 80 on dataset X.")
    doc.save(path)
    doc.close()
    with sessions.begin() as db:
        v = db.get(PaperVersion, "v1")
        v.status, v.attempt, v.sha256 = "parsing", 1, hashlib.sha256(path.read_bytes()).hexdigest()
        v.byte_size = path.stat().st_size
    ingest_version(sessions, settings, fake, "v1", 1)
    with sessions() as db:
        count = db.scalar(select(func.count()).select_from(Chunk))
        assert db.get(PaperVersion, "v1").status == "ready"
    with sessions.begin() as db:
        v = db.get(PaperVersion, "v1")
        v.status, v.attempt = "parsing", 2
    ingest_version(sessions, settings, fake, "v1", 2)
    with sessions() as db:
        assert db.scalar(select(func.count()).select_from(Chunk)) == count


def test_ingestion_parses_verified_copy_if_source_changes(sessions, settings, corpus, fake, monkeypatch):
    settings.storage_dir.mkdir()
    source = settings.storage_dir / "a.pdf"
    original = b"The original policy requires approval.\n"
    source.write_bytes(original)
    with sessions.begin() as db:
        version = db.get(PaperVersion, "v1")
        version.filename, version.status, version.attempt = "policy.txt", "parsing", 1
        version.sha256, version.byte_size = hashlib.sha256(original).hexdigest(), len(original)

    extract = ingestion.extract_document
    copies = []
    def replace_source_before_parse(path, filename, guard):
        copies.append(path)
        source.write_bytes(b"The replaced policy forbids approval.\n")
        return extract(path, filename, guard)

    monkeypatch.setattr(ingestion, "extract_document", replace_source_before_parse)
    ingest_version(sessions, settings, fake, "v1", 1)
    with sessions() as db:
        texts = list(db.scalars(select(SourceBlock.text)))
        assert any("original policy" in text for text in texts)
        assert not any("replaced policy" in text for text in texts)
    assert len(copies) == 1 and not copies[0].exists()


def test_ingestion_rejects_changed_source_before_parsing(sessions, settings, corpus, fake, monkeypatch):
    settings.storage_dir.mkdir()
    source = settings.storage_dir / "a.pdf"
    source.write_bytes(b"changed")
    with sessions.begin() as db:
        version = db.get(PaperVersion, "v1")
        version.filename, version.status, version.attempt = "policy.txt", "parsing", 1
        version.sha256, version.byte_size = hashlib.sha256(b"trusted").hexdigest(), len(b"trusted")
    monkeypatch.setattr(ingestion, "extract_document", lambda *_args: pytest.fail("must not parse changed content"))
    with pytest.raises(DomainError, match="不一致"):
        ingest_version(sessions, settings, fake, "v1", 1)
    with sessions() as db:
        assert db.get(PaperVersion, "v1").status == "parsing"
