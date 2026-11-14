"""Run offline chunk-structure checks or delegate live retrieval evaluation.

The live benchmark has one maintained entry point: scripts/benchmark_chunking.py.
Use --structural-only here to avoid external services and model charges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.chunking import TOKENIZER, ChunkPolicy, token_count  # noqa: E402
from app.ingestion import build_chunks, extract_document  # noqa: E402
from app.models import SourceBlock  # noqa: E402

FIXTURE_DIR = ROOT / "fixtures" / "chunking"
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
MAX_FIXTURE_DOCUMENTS = 8
MAX_FIXTURE_QUESTIONS = 32
MAX_FIXTURE_SOURCE_CHARS = 200_000
MAX_CHUNKS_PER_STRATEGY = 512


class EvaluationLimitError(RuntimeError):
    pass


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


def load_control_fixture():
    manifest_path = FIXTURE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("fictional") or manifest.get("human_reviewed"):
        raise ValueError("chunking control set must remain fictional and unreviewed")
    if len(manifest["documents"]) > MAX_FIXTURE_DOCUMENTS:
        raise EvaluationLimitError("control set exceeds the document limit")
    if len(manifest["questions"]) > MAX_FIXTURE_QUESTIONS:
        raise EvaluationLimitError("control set exceeds the question limit")

    blocks, raw_sections = [], {}
    document_hashes, document_chars = {}, 0
    for document in manifest["documents"]:
        path = FIXTURE_DIR / document["filename"]
        data = path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != document["sha256"]:
            raise ValueError(f"control file hash mismatch: {document['filename']}")
        raw, _, _ = extract_document(path, path.name)
        document_hashes[document["id"]] = actual_hash
        for item in raw:
            document_chars += len(item["text"])
            block_id = hashlib.sha256(
                f"chunk-control:{document['id']}:{item['page_index']}:{item['block_index']}".encode()
            ).hexdigest()[:32]
            source = SourceBlock(
                id=block_id,
                parse_revision_id="chunk-control-parse",
                page_index=item["page_index"],
                block_index=len(blocks),
                bbox=item["bbox"],
                text=item["text"],
                printed_page_label=item["printed_page_label"],
                section=f"{document['id']} / {item.get('section') or '正文'}",
            )
            blocks.append(source)
            raw_sections[block_id] = item.get("section") or "正文"
    if document_chars > MAX_FIXTURE_SOURCE_CHARS:
        raise EvaluationLimitError("control set exceeds the source character limit")

    questions = []
    for item in manifest["questions"]:
        expected = []
        for evidence in item["evidence"]:
            source_doc = evidence["document"]
            quote = evidence["quote"]
            matches = []
            for block in blocks:
                if not block.section.startswith(f"{source_doc} / "):
                    continue
                section = raw_sections[block.id]
                if evidence.get("section") and section != evidence["section"]:
                    continue
                start = block.text.find(quote)
                if start >= 0:
                    matches.append({
                        "block_id": block.id,
                        "char_start": start,
                        "char_end": start + len(quote),
                    })
            if not matches:
                raise ValueError(f"gold quote not found in source: {item['id']}")
            expected.extend(matches[:1])
        questions.append({
            "id": item["id"],
            "question": item["question"],
            "split": item["split"],
            "evidence": expected,
        })
    return manifest, blocks, questions, document_hashes, document_chars


def _policy(strategy: str) -> ChunkPolicy:
    return ChunkPolicy(strategy, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)


def _source_coverage(blocks, chunks):
    ranges_by_block: dict[str, list[tuple[int, int]]] = {block.id: [] for block in blocks}
    requested_ranges = 0
    for chunk in chunks:
        for span in chunk["source_ranges"]:
            ranges_by_block.setdefault(span["block_id"], []).append(
                (span["char_start"], span["char_end"])
            )
            requested_ranges += span["char_end"] - span["char_start"]

    unique_ranges = 0
    covered_nonspace = 0
    total_nonspace = 0
    for block in blocks:
        text_value = block.text
        total_nonspace += sum(not char.isspace() for char in text_value)
        merged = []
        for start, end in sorted(ranges_by_block.get(block.id, [])):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in merged:
            unique_ranges += end - start
            covered_nonspace += sum(not char.isspace() for char in text_value[start:end])
    return {
        "nonspace_source_coverage": _ratio(covered_nonspace, total_nonspace),
        "overlap_source_chars": max(0, requested_ranges - unique_ranges),
        "requested_source_chars": requested_ranges,
    }


def _gold_span_metrics(questions, chunks):
    ranges = [span for chunk in chunks for span in chunk["source_ranges"]]
    answer = {}
    for split in ("all", "development", "held_out"):
        selected = questions if split == "all" else [q for q in questions if q["split"] == split]
        total, found, complete = 0, 0, 0
        for question in selected:
            question_found = 0
            for expected in question["evidence"]:
                total += 1
                if any(
                    span["block_id"] == expected["block_id"]
                    and span["char_start"] <= expected["char_start"]
                    and span["char_end"] >= expected["char_end"]
                    for span in ranges
                ):
                    found += 1
                    question_found += 1
            if question["evidence"] and question_found == len(question["evidence"]):
                complete += 1
        answer[split] = {
            "evidence_spans_found": found,
            "evidence_spans_total": total,
            "gold_quote_containment": _ratio(found, total),
            "complete_questions": complete,
            "questions_total": len(selected),
            "complete_question_containment": _ratio(complete, len(selected)),
        }
    return answer


def structural_metrics(blocks, questions, chunks):
    token_lengths = [token_count(chunk["text"]) for chunk in chunks]
    char_lengths = [len(chunk["text"]) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "tokens": {
            "p50": _percentile(token_lengths, .50),
            "p95": _percentile(token_lengths, .95),
            "max": max(token_lengths, default=0),
        },
        "characters": {
            "p50": _percentile(char_lengths, .50),
            "p95": _percentile(char_lengths, .95),
            "max": max(char_lengths, default=0),
        },
        "exact_duplicate_chunks": len(chunks) - len({chunk["text"] for chunk in chunks}),
        **_source_coverage(blocks, chunks),
        "gold_containment": _gold_span_metrics(questions, chunks),
    }


def run_structural_comparison():
    manifest, blocks, questions, hashes, source_chars = load_control_fixture()
    page_heights = {block.page_index: 842 for block in blocks}
    results = {}
    for strategy in ("legacy", "fixed", "paragraph", "structure"):
        chunks = build_chunks(blocks, page_heights, policy=_policy(strategy))
        if len(chunks) > MAX_CHUNKS_PER_STRATEGY:
            raise EvaluationLimitError(f"{strategy} exceeds the chunk count limit")
        results[strategy] = structural_metrics(blocks, questions, chunks)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "fictional long-document control set; structural and exact gold-span checks only",
        "fixture": {
            "documents": len(manifest["documents"]),
            "questions": len(questions),
            "source_chars": source_chars,
            "sha256": hashes,
            "human_reviewed": manifest.get("human_reviewed", False),
        },
        "parameters": {
            "tokenizer": TOKENIZER,
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        },
        "strategies": results,
        "skipped": {"semantic": "requires the configured Embedding service"},
        "retrieval": None,
        "limitations": [
            "No embedding service, PostgreSQL, Milvus, chat model, or reranker was used.",
            "Gold-span containment does not measure semantic retrieval Recall@K or answer quality.",
            "The fictional control set is not human-reviewed and is not a deployment recommendation.",
        ],
    }


def write_report(path: Path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="compare source coverage and exact gold spans without external services",
    )
    parser.add_argument("--output", type=Path, help="report path for offline or delegated evaluation")
    args, forwarded = parser.parse_known_args(argv)

    if args.structural_only:
        if forwarded:
            parser.error(f"unsupported structural-only arguments: {' '.join(forwarded)}")
        result = run_structural_comparison()
        output = args.output or ROOT / "docs" / "chunk-strategy-structural-evaluation.json"
        write_report(output, result)
        print(json.dumps({"report": str(output), "strategies": list(result["strategies"])}, ensure_ascii=False))
        return 0

    script = ROOT / "scripts" / "benchmark_chunking.py"
    command = [sys.executable, str(script), *forwarded]
    if args.output:
        command.extend(["--output", str(args.output)])
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
