import codecs
import csv
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pymupdf
from sqlalchemy import select, update

from app.core.errors import BudgetExhausted, DomainError, InterruptedRun, ProviderError
from app.core.models import Chunk, IndexRevision, Paper, PaperVersion, ParseRevision, SourceBlock, uid
from app.knowledge.chunking import VERSION as CHUNK_CONFIG
from app.knowledge.chunking import ChunkPolicy, split_blocks
from app.knowledge.llama_rag import embed_chunks, index_chunks
from app.knowledge.vector_store import collection_name, production_store

PARSE_CONFIG = f"multiformat-v5-pymupdf-{pymupdf.VersionBind}"
SUPPORTED_SUFFIXES = frozenset({
    ".pdf", ".docx", ".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".jsonl",
    ".html", ".htm", ".xlsx",
})
MAX_STRUCTURED_ROWS = 100_000
MAX_STRUCTURED_COLUMNS = 1_000
MAX_JSON_VALUES = 100_000
MAX_JSON_DEPTH = 64
MAX_XLSX_SHEETS = 100
MAX_XLSX_ROWS = 100_000
MAX_XLSX_TOTAL_CELLS = 2_000_000
MAX_XLSX_EXPANDED_BYTES = 80 * 1024 * 1024


class _VisibleHTML(HTMLParser):
    hidden_tags = {"script", "style", "noscript", "svg", "template"}
    block_tags = {
        "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "figure",
        "footer", "form", "header", "hr", "li", "main", "nav", "ol", "p", "section", "table", "ul",
    }
    heading_tags = {f"h{i}" for i in range(1, 7)}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.section = "正文"
        self.parts = []
        self.heading_tag = None
        self.heading_parts = []
        self.hidden_depth = 0
        self.hidden_stack = []

    def flush(self):
        text = "".join(self.parts).strip()
        if text:
            self.blocks.append((self.section, text))
        self.parts.clear()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        style = re.sub(r"\s+", "", attributes.get("style", "").lower())
        hidden = (tag in self.hidden_tags or "hidden" in attributes or
                  attributes.get("aria-hidden", "").lower() == "true" or
                  "display:none" in style or "visibility:hidden" in style)
        if hidden:
            self.hidden_depth += 1
            self.hidden_stack.append(tag)
            return
        if self.hidden_depth:
            return
        if tag in self.heading_tags:
            self.flush()
            self.heading_tag, self.heading_parts = tag, []
        elif tag in self.block_tags:
            self.flush()
        if tag in {"td", "th"} and self.parts and not self.parts[-1].endswith(" | "):
            self.parts.append(" | ")
        if tag == "img":
            alt = dict(attrs).get("alt", "").strip()
            if alt:
                self.parts.append(f" {alt} ")

    def handle_endtag(self, tag):
        if self.hidden_depth:
            for index in range(len(self.hidden_stack) - 1, -1, -1):
                if self.hidden_stack[index] == tag:
                    del self.hidden_stack[index:]
                    self.hidden_depth = len(self.hidden_stack)
                    return
            return
        if self.heading_tag == tag:
            text = " ".join("".join(self.heading_parts).split())
            self.heading_tag, self.heading_parts = None, []
            if text:
                self.section = text[:500]
                self.blocks.append((self.section, text))
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag in self.block_tags:
            self.flush()

    def handle_data(self, data):
        if self.hidden_depth:
            return
        if self.heading_tag:
            self.heading_parts.append(data)
        else:
            self.parts.append(data)


def index_config(settings) -> str:
    fingerprint = hashlib.sha256(json.dumps({
        "base_url": settings.embedding_base_url.rstrip("/"), "model": settings.embedding_model,
        "dimension": settings.embedding_dimensions, "revision": settings.embedding_revision,
        "chunk_policy": ChunkPolicy.from_settings(settings).manifest(),
    }, sort_keys=True).encode()).hexdigest()[:16]
    return f"{CHUNK_CONFIG}:{fingerprint}"


def _pdf_table_blocks(page, page_index: int, printed_page_label, page_height: float, guard=None
                      ) -> tuple[list[dict], bool]:
    """Return structured rows for clearly ruled PDF tables and their bounding boxes."""
    if guard:
        guard()
    try:
        tables = page.find_tables(strategy="lines").tables
    except Exception:
        # Table detection is an enhancement; normal text extraction remains the fallback.
        return [], True
    if guard:
        guard()

    groups, extraction_failed = [], False
    for table_number, table in enumerate(tables, start=1):
        if guard:
            guard()
        try:
            rows = table.extract()
        except Exception:
            # Keep the original text blocks if a detected table cannot be extracted.
            extraction_failed = True
            continue
        if len(rows) < 2 or not rows[0]:
            continue
        if len(rows) > MAX_STRUCTURED_ROWS or len(rows[0]) > MAX_STRUCTURED_COLUMNS:
            continue

        header_index = next((i for i, row in enumerate(rows) if row and any(
            value is not None and str(value).strip() for value in row)), None)
        if header_index is None or header_index == len(rows) - 1:
            continue
        header_row = rows[header_index]
        headers = [str(value).strip() if value is not None and str(value).strip() else f"列{i + 1}"
                   for i, value in enumerate(header_row)]
        table_box = tuple(float(value) for value in table.bbox)
        section_prefix = f"PDF 第 {page_index + 1} 页 · 表格 {table_number}"
        row_blocks = []

        for row_index, cells in enumerate(rows[header_index + 1:], start=header_index + 2):
            if guard:
                guard()
            values = ["" if value is None else str(value).strip() for value in cells]
            while values and not values[-1]:
                values.pop()
            fields = [f"{headers[i] if i < len(headers) else f'列{i + 1}'}: {value}"
                      for i, value in enumerate(values) if value]
            if not fields:
                continue
            row_bbox = table.rows[row_index - 1].bbox
            row_blocks.append({
                "page_index": page_index,
                "bbox": [float(value) for value in row_bbox],
                "text": " | ".join(fields),
                "printed_page_label": printed_page_label,
                "page_height": page_height,
                "section": f"{section_prefix} · 第 {row_index} 行",
            })
        if row_blocks:
            groups.append({"bbox": table_box, "blocks": row_blocks})
    return groups, extraction_failed


def _bbox_contains(outer, inner, tolerance=2.0):
    return (inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance and
            inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def extract_pdf(path: Path, guard=None) -> tuple[list[dict], int, list[str]]:
    try:
        with pymupdf.open(path) as doc:
            if not doc.is_pdf or doc.needs_pass:
                raise DomainError("仅支持未加密、含文字层的 PDF")
            if not 1 <= len(doc) <= 1000:
                raise DomainError("PDF 页数需要在 1–1000 之间")
            blocks, empty_pages, table_detection_failed = [], [], False
            for page_index, page in enumerate(doc):
                if guard:
                    guard()
                count = 0
                page_text_blocks = []
                for block in page.get_text("blocks", sort=False):
                    if len(block) >= 7 and block[6] != 0:
                        continue
                    text = block[4]
                    if not text.strip():
                        continue
                    count += len(text.strip())
                    page_text_blocks.append({"page_index": page_index,
                                             "bbox": [float(v) for v in block[:4]], "text": text,
                                             "printed_page_label": page.get_label() or None,
                                             "page_height": page.rect.height})

                table_groups, failed = _pdf_table_blocks(
                    page, page_index, page.get_label() or None, page.rect.height, guard)
                table_detection_failed = table_detection_failed or failed
                emitted_tables = set()
                page_blocks = []
                for text_block in page_text_blocks:
                    matches = [i for i, group in enumerate(table_groups)
                               if _bbox_contains(group["bbox"], text_block["bbox"])]
                    if matches:
                        for table_index in matches:
                            if table_index not in emitted_tables:
                                page_blocks.extend(table_groups[table_index]["blocks"])
                                emitted_tables.add(table_index)
                        continue
                    page_blocks.append(text_block)
                # A detected table can contain vector text that does not appear in text blocks.
                for table_index, group in enumerate(table_groups):
                    if table_index not in emitted_tables:
                        page_blocks.extend(group["blocks"])
                for block in page_blocks:
                    block["block_index"] = len(blocks)
                    blocks.append(block)
                if count < 20:
                    empty_pages.append(page_index + 1)
            if not blocks or sum(len(b["text"].strip()) for b in blocks) < 40:
                raise DomainError("没有足够的可提取文字；扫描件需 OCR，首版不支持")
            warnings = ["PDF 保留内容流顺序；可检测的有边框表格按表头和行结构提取。双栏、无边框表格、公式及扫描件 OCR 仍需对照原文件。"]
            if table_detection_failed:
                warnings.append("部分 PDF 页面表格结构提取失败，已回退为普通文本块。")
            if empty_pages:
                warnings.append("以下 PDF 页文字不足，检索覆盖可能受限：" + ", ".join(map(str, empty_pages)))
            return blocks, len(doc), warnings
    except DomainError:
        raise
    except (BudgetExhausted, InterruptedRun):
        raise
    except Exception:
        raise DomainError("PDF 损坏或无法解析") from None


def clean(text: str) -> str:
    text = re.sub(r"([A-Za-z])-\r?\n([a-z])", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def _text_bytes(path: Path, kind: str) -> str:
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return data.decode("utf-16")
        except UnicodeError:
            raise DomainError(f"{kind} 文本编码无效") from None
    if b"\x00" in data:
        raise DomainError(f"{kind} 不能包含二进制内容")
    try:
        return data.decode("utf-8-sig")
    except UnicodeError:
        try:
            return data.decode("gb18030")
        except UnicodeError:
            raise DomainError(f"{kind} 需要使用 UTF-8、UTF-16 或 GB18030 编码") from None


def _table_rows(rows, section_prefix: str, guard=None, allow_empty=False):
    raw = []
    headers = None
    for row_number, cells in enumerate(rows, start=1):
        if guard:
            guard()
        if row_number > MAX_STRUCTURED_ROWS:
            raise DomainError(f"结构化表格超过 {MAX_STRUCTURED_ROWS} 行")
        values = ["" if value is None else str(value).strip() for value in cells]
        while values and not values[-1]:
            values.pop()
        if not any(values):
            continue
        if len(values) > MAX_STRUCTURED_COLUMNS:
            raise DomainError(f"结构化表格超过 {MAX_STRUCTURED_COLUMNS} 列")
        if headers is None:
            headers = [value or f"列{i + 1}" for i, value in enumerate(values)]
            section = f"{section_prefix} · 表头"
            raw.append({"page_index": 0, "block_index": len(raw), "bbox": [],
                        "text": " | ".join(headers), "printed_page_label": None,
                        "page_height": 842, "section": section})
            continue
        fields = [f"{headers[i] if i < len(headers) else f'列{i + 1}'}: {value}"
                  for i, value in enumerate(values) if value]
        if fields:
            section = f"{section_prefix} · 第 {row_number} 行"
            raw.append({"page_index": 0, "block_index": len(raw), "bbox": [],
                        "text": " | ".join(fields), "printed_page_label": None,
                        "page_height": 842, "section": section})
    if not raw:
        if allow_empty:
            return []
        raise DomainError("表格没有可索引的数据行")
    return raw


def _json_blocks(value, guard=None):
    blocks, visited = [], 0

    def visit(item, path, depth):
        nonlocal visited
        if guard:
            guard()
        visited += 1
        if visited > MAX_JSON_VALUES:
            raise DomainError(f"JSON 节点超过 {MAX_JSON_VALUES} 个")
        if depth > MAX_JSON_DEPTH:
            raise DomainError(f"JSON 嵌套深度超过 {MAX_JSON_DEPTH}")
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key), depth + 1)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", depth + 1)
        else:
            if item is None:
                return
            text = json.dumps(item, ensure_ascii=False) if isinstance(item, (bool, int, float)) else str(item)
            if not text.strip():
                return
            section = path or "根值"
            blocks.append({"page_index": 0, "block_index": len(blocks), "bbox": [], "text": text,
                           "printed_page_label": None, "page_height": 842, "section": section[:500]})

    visit(value, "", 0)
    if not blocks:
        raise DomainError("JSON 没有可索引的标量数据")
    return blocks


def _extract_json(path: Path, json_lines: bool, guard=None):
    content = _text_bytes(path, "JSON")
    try:
        if not json_lines:
            return _json_blocks(json.loads(content), guard)
        raw, total_values = [], 0
        for line_number, line in enumerate(content.splitlines(), start=1):
            if guard:
                guard()
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise DomainError(f"JSONL 第 {line_number} 行格式错误") from None
            for block in _json_blocks(row, guard):
                if guard:
                    guard()
                total_values += 1
                if total_values > MAX_JSON_VALUES:
                    raise DomainError(f"JSONL 节点超过 {MAX_JSON_VALUES} 个")
                block["section"] = f"第 {line_number} 行 · {block['section']}"
                block["block_index"] = len(raw)
                raw.append(block)
        if not raw:
            raise DomainError("JSONL 没有可索引的数据")
        return raw
    except DomainError:
        raise
    except (json.JSONDecodeError, RecursionError):
        raise DomainError("JSON 格式错误或嵌套过深") from None


def _extract_spreadsheet(path: Path, guard=None):
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            if sum(info.file_size for info in infos) > MAX_XLSX_EXPANDED_BYTES:
                raise DomainError("XLSX 解压后内容超过 80 MB")
            if any(info.file_size > 1024 * 1024 and info.compress_size == 0 or
                   info.compress_size and info.file_size / info.compress_size > 200
                   for info in infos):
                raise DomainError("XLSX 压缩比例过高")
    except DomainError:
        raise
    except Exception:
        raise DomainError("XLSX 文件无法读取") from None

    from openpyxl import load_workbook

    try:
        workbook = load_workbook(path, read_only=True, data_only=False)
        raw, total_cells = [], 0
        try:
            if not 1 <= len(workbook.worksheets) <= MAX_XLSX_SHEETS:
                raise DomainError(f"XLSX 工作表数量需要在 1–{MAX_XLSX_SHEETS} 之间")
            for sheet in workbook.worksheets:
                if guard:
                    guard()
                if sheet.max_row > MAX_XLSX_ROWS or sheet.max_column > MAX_STRUCTURED_COLUMNS:
                    raise DomainError("XLSX 工作表超过行数或列数限制")
                total_cells += sheet.max_row * sheet.max_column
                if total_cells > MAX_XLSX_TOTAL_CELLS:
                    raise DomainError("XLSX 单元格总数超过限制")
                section = f"工作表：{sheet.title}"
                sheet_blocks = _table_rows(sheet.iter_rows(values_only=True), section, guard, allow_empty=True)
                for block in sheet_blocks:
                    block["block_index"] = len(raw)
                    raw.append(block)
        finally:
            workbook.close()
    except DomainError:
        raise
    except (BudgetExhausted, InterruptedRun):
        raise
    except Exception:
        raise DomainError("XLSX 文件损坏或包含不支持的内容") from None
    if not raw:
        raise DomainError("XLSX 没有可索引的数据")
    return raw


def extract_document(path: Path, filename: str, guard=None):
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, guard)
    raw, section = [], "正文"

    def add(text):
        if text.strip():
            raw.append({"page_index": 0, "block_index": len(raw), "bbox": [], "text": text,
                        "printed_page_label": None, "page_height": 842, "section": section})

    if suffix in (".md", ".markdown"):
        content = _text_bytes(path, "Markdown")
        if "\x00" in content:
            raise DomainError("Markdown 不能包含二进制内容")
        paragraph, fenced = [], False
        for line in content.splitlines(keepends=True):
            if guard:
                guard()
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
            if not fenced and re.match(r"^#{1,6}\s+", line):
                add("".join(paragraph))
                paragraph = []
                section = re.sub(r"^#{1,6}\s+", "", line).strip()[:500]
                add(line)
            elif not line.strip() and not fenced:
                add("".join(paragraph))
                paragraph = []
            else:
                paragraph.append(line)
        add("".join(paragraph))
    elif suffix == ".docx":
        from docx import Document
        from docx.table import Table
        try:
            with ZipFile(path) as archive:
                if sum(i.file_size for i in archive.infolist()) > 80 * 1024 * 1024:
                    raise DomainError("Word 解压后内容超过 80 MB")
            doc = Document(path)
            for item in doc.iter_inner_content():
                if guard:
                    guard()
                if isinstance(item, Table):
                    add("\n".join(" | ".join(c.text for c in row.cells) for row in item.rows))
                else:
                    if item.style and item.style.name.startswith("Heading"):
                        section = item.text.strip()[:500] or section
                    add(item.text + "\n")
        except (BadZipFile, KeyError, ValueError):
            raise DomainError("Word 文件无法解析，请上传有效的 .docx") from None
    elif suffix in (".txt", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".xlsx"):
        if suffix == ".txt":
            content = _text_bytes(path, "TXT")
            paragraph = []
            for line in content.splitlines(keepends=True):
                if guard:
                    guard()
                if not line.strip():
                    add("".join(paragraph))
                    paragraph = []
                else:
                    paragraph.append(line)
            add("".join(paragraph))
        elif suffix in (".csv", ".tsv"):
            content = _text_bytes(path, "CSV/TSV")
            try:
                rows = csv.reader(StringIO(content, newline=""), delimiter="\t" if suffix == ".tsv" else ",",
                                  strict=True)
                raw = _table_rows(rows, Path(filename).name, guard)
            except DomainError:
                raise
            except csv.Error:
                raise DomainError("CSV/TSV 行格式错误") from None
        elif suffix in (".json", ".jsonl"):
            raw = _extract_json(path, suffix == ".jsonl", guard)
        elif suffix in (".html", ".htm"):
            content = _text_bytes(path, "HTML")
            parser = _VisibleHTML()
            try:
                for start in range(0, len(content), 4096):
                    if guard:
                        guard()
                    parser.feed(content[start:start + 4096])
                parser.close()
                parser.flush()
            except (ValueError, AssertionError):
                raise DomainError("HTML 内容无法解析") from None
            for section, text in parser.blocks:
                raw.append({"page_index": 0, "block_index": len(raw), "bbox": [], "text": text,
                            "printed_page_label": None, "page_height": 842, "section": section})
        else:
            raw = _extract_spreadsheet(path, guard)
    else:
        raise DomainError("支持 PDF、DOCX、Markdown、TXT、CSV/TSV、JSON/JSONL、HTML 和 XLSX")
    if not raw:
        raise DomainError("文档没有可提取的文字")
    warnings = ["非 PDF 文件按章节、表格行或字段路径定位；不虚构页码。"]
    if suffix == ".xlsx":
        warnings.append("XLSX 公式按公式文本索引，不计算公式；隐藏工作表也会纳入索引。")
    if suffix in (".html", ".htm"):
        warnings.append("HTML 静态提取文本；脚本、样式、模板及带 hidden/aria-hidden/常见行内隐藏样式的元素不纳入检索。")
    if suffix in (".csv", ".tsv"):
        warnings.append("CSV/TSV 第一条非空记录按列名处理；后续引用位置按源文件行号显示。")
    return raw, None, warnings


def build_chunks(blocks: list[SourceBlock], page_heights: dict[int, float], max_chars=None, guard=None,
                 *, policy=None, provider=None, model_name="") -> list[dict]:
    policy = policy or ChunkPolicy()
    if policy.strategy != "legacy" and max_chars is None:
        return split_blocks(blocks, page_heights, policy, provider=provider, model_name=model_name, guard=guard)
    return legacy_chunks(blocks, page_heights, max_chars or 3200, guard)


def legacy_chunks(blocks: list[SourceBlock], page_heights: dict[int, float], max_chars=3200, guard=None) -> list[dict]:
    # Repeated margin text is excluded from search, never removed from SourceBlock.
    def is_margin(block):
        height = page_heights.get(block.page_index, 842)
        return len(block.bbox) >= 4 and (block.bbox[3] < height * .09 or block.bbox[1] > height * .91)

    margin_pages = defaultdict(set)
    for block in blocks:
        if guard:
            guard()
        if is_margin(block):
            margin_pages[clean(block.text)].add(block.page_index)
    chunks, current, current_size, page = [], [], 0, -1

    def emit():
        if current:
            chunks.append({"text": "\n\n".join(clean(raw) for _, _, _, raw in current),
                           "source_ranges": [{"block_id": bid, "char_start": start, "char_end": end}
                                             for bid, start, end, _ in current]})

    for block in blocks:
        if guard:
            guard()
        if is_margin(block) and len(margin_pages[clean(block.text)]) >= 3:
            continue
        # Source offsets always refer to the unmodified raw text.
        for start in range(0, len(block.text), 2200):
            raw = block.text[start:start + 2200]
            if not raw.strip():
                continue
            location = (block.page_index, block.section)
            if current and (page != location or current_size + len(raw) > max_chars):
                emit()
                previous = current[-1:] if page == location and len(current[-1][3]) < 800 else []
                current = previous
                current_size = sum(len(t[3]) for t in current)
            page = location
            current.append((block.id, start, start + len(raw), raw))
            current_size += len(raw)
    emit()
    return chunks


def ingest_version(session_factory, settings, provider, version_id: str, attempt: int, guard=None,
                   vector_store=None):
    check = guard or (lambda: None)
    check()
    with session_factory() as db:
        version = db.get(PaperVersion, version_id)
        if not version or version.attempt != attempt or version.status != "parsing":
            raise InterruptedRun()
        path = settings.storage_dir / version.storage_key
        filename, expected_hash, expected_size = version.filename, version.sha256, version.byte_size
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DomainError("不支持的文件格式")
    # Parse the same private copy that was hashed, rather than reopening the
    # original path after verification. The directory is removed on all exits.
    with tempfile.TemporaryDirectory(prefix="knowledge-ingest-") as temporary_dir:
        verified_path = Path(temporary_dir) / ("source" + suffix)
        checksum, size = hashlib.sha256(), 0
        with path.open("rb") as source, verified_path.open("wb") as copy:
            while chunk := source.read(1024 * 1024):
                check()
                size += len(chunk)
                if size > expected_size:
                    raise DomainError("文件哈希或大小与版本不一致，停止入库")
                checksum.update(chunk)
                copy.write(chunk)
        if size != expected_size or checksum.hexdigest() != expected_hash:
            raise DomainError("文件哈希或大小与版本不一致，停止入库")
        check()
        raw, page_count, warnings = extract_document(verified_path, filename, check)
    check()
    # Deterministic IDs make retries idempotent without overwriting an old parse.
    parse_id = hashlib.sha256(f"{version_id}:{PARSE_CONFIG}".encode()).hexdigest()[:32]
    blocks = [SourceBlock(id=hashlib.sha256(f"{parse_id}:{b['page_index']}:{b['block_index']}".encode()).hexdigest()[:32],
                          parse_revision_id=parse_id, **{k: v for k, v in b.items() if k != "page_height"}) for b in raw]
    chunks = build_chunks(blocks, {b["page_index"]: b["page_height"] for b in raw}, guard=check,
                          policy=ChunkPolicy.from_settings(settings), provider=provider,
                          model_name=settings.embedding_model)
    if not chunks:
        raise DomainError("没有可索引片段")
    with session_factory.begin() as db:
        changed = db.execute(update(PaperVersion).where(
            PaperVersion.id == version_id, PaperVersion.attempt == attempt, PaperVersion.status == "parsing",
        ).values(status="indexing"))
        if not changed.rowcount:
            raise InterruptedRun()
    vectors = embed_chunks(chunks, provider, settings.embedding_model, check)
    with session_factory.begin() as db:
        # Match the publication lock order: document before its versions.
        # Recheck the association after locking both rows.
        paper_id = db.scalar(select(PaperVersion.paper_id).where(PaperVersion.id == version_id))
        paper = db.scalar(select(Paper).where(Paper.id == paper_id).with_for_update()) if paper_id else None
        version = db.scalar(select(PaperVersion).where(PaperVersion.id == version_id).with_for_update())
        if (not version or not paper or version.paper_id != paper.id or version.attempt != attempt or
                version.status != "indexing" or paper.deleted):
            raise InterruptedRun()
        parse = db.get(ParseRevision, parse_id)
        if not parse:
            db.add(ParseRevision(id=parse_id, paper_version_id=version_id,
                                 parser=f"PyMuPDF {pymupdf.VersionBind}", config_version=PARSE_CONFIG, status="ready"))
            db.flush()
            db.add_all(blocks)
            db.flush()
        existing = db.scalar(select(IndexRevision).where(
            IndexRevision.parse_revision_id == parse_id, IndexRevision.config_version == index_config(settings),
            IndexRevision.embedding_model == settings.embedding_model,
        ))
        if not existing:
            index = IndexRevision(id=uid(), parse_revision_id=parse_id, config_version=index_config(settings),
                                  embedding_model=settings.embedding_model, embedding_dimensions=len(vectors[0]),
                                  vector_backend=(settings.vector_store if db.bind.dialect.name == "postgresql"
                                                  else "sqlite_test"),
                                  vector_collection=None,
                                  status="ready")
            if db.bind.dialect.name == "postgresql":
                index.vector_collection = collection_name(index.id, settings.milvus_collection_prefix)
            db.add(index)
            db.flush()
            # PostgreSQL keeps source text and FTS fields; production vectors live in Milvus.
            stored_embeddings = vectors if db.bind.dialect.name != "postgresql" else [None] * len(vectors)
            chunk_rows = [Chunk(index_revision_id=index.id, embedding=v, **c)
                          for c, v in zip(chunks, stored_embeddings)]
            db.add_all(chunk_rows)
            db.flush()
            if db.bind.dialect.name == "postgresql":
                owned_store = vector_store is None
                store = vector_store or production_store(settings, db)
                try:
                    index_chunks(db, index, chunk_rows, vectors, store, provider, check)
                finally:
                    if owned_store:
                        store.close()
            if db.bind.dialect.name == "postgresql":
                from sqlalchemy import func
                db.execute(update(Chunk).where(Chunk.index_revision_id == index.id).values(
                    search_vector=func.to_tsvector("simple", Chunk.text)))
        elif db.bind.dialect.name == "postgresql" and (
                existing.vector_backend != "milvus" or not existing.vector_collection):
            raise ProviderError("已有索引尚未迁入 Milvus，请先运行 migrate-vectors", "vector_store_config")
        check()
        version.status = "ready"
        # Publish the replacement only when parsing and indexing both succeeded.
        # A failed update leaves the previous version available.
        db.flush()
        newest_ready = db.scalar(select(PaperVersion.id).where(
            PaperVersion.paper_id == paper.id, PaperVersion.status == "ready").order_by(
            PaperVersion.created_at.desc(), PaperVersion.id.desc()).limit(1))
        if newest_ready == version.id:
            db.execute(update(PaperVersion).where(PaperVersion.paper_id == paper.id,
                                                 PaperVersion.id != version.id).values(is_active=False))
            version.is_active = True
        else:
            version.is_active = False
        version.page_count = page_count
        version.warnings = warnings
        version.error = None
        version.lease_until = None
