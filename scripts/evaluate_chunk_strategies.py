"""Compare chunk layouts against the project's explicitly fictional PDF gold lines.

The output measures segmentation and verbatim evidence coverage only. It does
not call an embedding/chat service or claim retrieval-quality improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ingestion import clean, extract_document, legacy_chunks  # noqa: E402
from app.models import SourceBlock  # noqa: E402


def normalized(text: str) -> str:
    return " ".join(clean(text).casefold().split())


def source_block_chunks(blocks: list[SourceBlock], max_chars: int = 2200) -> list[dict]:
    """Treat each extracted PDF paragraph/block as a chunk boundary, without overlap."""
    chunks = []
    for block in blocks:
        for start in range(0, len(block.text), max_chars):
            end = min(start + max_chars, len(block.text))
            text = clean(block.text[start:end])
            if text:
                chunks.append({"text": text, "source_ranges": [{
                    "block_id": block.id, "char_start": start, "char_end": end,
                }]})
    return chunks


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def measure(chunks: list[dict], evidence_lines: list[str]) -> dict:
    texts = [normalized(chunk["text"]) for chunk in chunks]
    lengths = [len(text) for text in texts]
    found = [line for line in evidence_lines if any(normalized(line) in text for text in texts)]
    evidence_hit_counts = [sum(normalized(line) in text for text in texts) for line in evidence_lines]
    duplicates = sum(count - 1 for count in Counter(text for text in texts if text).values() if count > 1)
    return {
        "chunks": len(chunks),
        "chunk_chars": {"p50": percentile(lengths, .50), "p95": percentile(lengths, .95),
                        "max": max(lengths, default=0)},
        "gold_lines": len(evidence_lines),
        "gold_lines_wholly_present": len(found),
        "gold_line_coverage": len(found) / len(evidence_lines) if evidence_lines else None,
        "gold_line_chunk_matches": sum(evidence_hit_counts),
        "exact_duplicate_chunks": duplicates,
    }


def run() -> dict:
    manifest = json.loads((ROOT / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("fictional") is not True:
        raise ValueError("This evaluation is intentionally restricted to the fictional fixture manifest")
    strategies = {
        "production_section_page_3200": {},
        "section_page_800": {"max_chars": 800},
        "source_block_2200_no_overlap": None,
    }
    documents = {}
    for paper in manifest["papers"]:
        path = ROOT / "fixtures" / paper["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != paper["sha256"]:
            raise ValueError(f"Fixture hash mismatch: {paper['filename']}")
        raw, _page_count, _warnings = extract_document(path, paper["filename"])
        blocks = [SourceBlock(id=f"{paper['filename']}:{item['page_index']}:{item['block_index']}",
                              parse_revision_id=paper["filename"],
                              **{key: value for key, value in item.items() if key != "page_height"})
                  for item in raw]
        page_heights = {item["page_index"]: item["page_height"] for item in raw}
        document_result = {}
        for name, config in strategies.items():
            chunks = (source_block_chunks(blocks) if config is None else
                      legacy_chunks(blocks, page_heights, **config))
            document_result[name] = measure(chunks, paper["verbatim_lines"])
        documents[paper["filename"]] = document_result
    totals = {}
    for name in strategies:
        per_doc = [documents[paper["filename"]][name] for paper in manifest["papers"]]
        gold_count = sum(item["gold_lines"] for item in per_doc)
        gold_found = sum(item["gold_lines_wholly_present"] for item in per_doc)
        lengths = [item["chunk_chars"]["p50"] for item in per_doc if item["chunk_chars"]["p50"] is not None]
        totals[name] = {
            "documents": len(per_doc),
            "chunks": sum(item["chunks"] for item in per_doc),
            "gold_lines": gold_count,
            "gold_lines_wholly_present": gold_found,
            "gold_line_coverage": gold_found / gold_count if gold_count else None,
            "exact_duplicate_chunks": sum(item["exact_duplicate_chunks"] for item in per_doc),
            "median_of_document_chunk_p50_chars": statistics.median(lengths) if lengths else None,
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "three SHA-256-verified fictional teaching PDFs; segmentation and exact gold-line coverage only",
        "limitations": [
            "No production or user documents were included.",
            "No embeddings, database retrieval, chat model, or reranker were called.",
            "Gold-line coverage is not Recall@K, answer quality, or a deployment recommendation.",
        ],
        "totals": totals,
        "documents": documents,
        "reranker": {
            "production_candidate_pool_limit": 24,
            "contract_test": "backend/tests/test_team.py::test_reranker_selects_actual_result_blocks_instead_of_repeated_headings",
            "real_model_pairwise_effect": "not measured; requires an approved model run and a reviewed query set",
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/chunk-strategy-fixture-evaluation.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
