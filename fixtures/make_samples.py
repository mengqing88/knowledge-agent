"""Create the document's explicitly fictional teaching PDFs, never real research claims."""
import hashlib
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
PAPERS = [
    ("teaching-a.pdf", "Fictional Paper A - Accuracy on dataset X", 4, [
        "On dataset X, method M achieved an F1 score of 82.",
        "The baseline achieved an F1 score of 80 under the same evaluation setting.",
        "These results only describe dataset X and the reported baseline.",
    ]),
    ("teaching-b.pdf", "Fictional Paper B - Accuracy and inference latency", 6, [
        "On dataset Y, method M and the baseline both achieved an F1 score of 75.",
        "Under this evaluation setting, average inference latency was 120 ms for M.",
        "The baseline average inference latency was 100 ms in the same setting.",
        "The dataset and evaluation setting differ from Fictional Paper A.",
    ]),
    ("teaching-c.pdf", "Fictional Paper C - Architecture only", 3, [
        "Method M contains an encoder, a retrieval module, and a prediction head.",
        "This teaching document describes the architecture only.",
        "It does not report GPU memory measurements or a matched latency experiment.",
    ]),
]


def main():
    manifest = {"fictional": True, "purpose": "Control-flow and citation demonstrations only", "papers": []}
    for filename, title, pages, facts in PAPERS:
        path = ROOT / filename
        doc = canvas.Canvas(str(path), pagesize=A4)
        doc.setTitle(title)
        for page_number in range(1, pages + 1):
            doc.setFillColor(colors.HexColor("#216c59"))
            doc.setFont("Helvetica-Bold", 10)
            doc.drawString(56, 786, "FICTIONAL TEACHING MATERIAL / NOT A REAL PAPER")
            doc.setFillColor(colors.HexColor("#202d2a"))
            doc.setFont("Helvetica-Bold", 16)
            doc.drawString(56, 738, title.split(" - ")[0])
            doc.setFont("Helvetica", 11)
            doc.drawString(56, 708, title.split(" - ")[1])
            y = 644
            lines = facts if page_number == pages else [
                "This page contains fictional background material for a software demonstration.",
                "All experiment details for this teaching paper are on its final PDF page.",
                "These files must not be cited as actual research or evaluation results.",
            ]
            for line in lines:
                doc.drawString(56, y, line)
                y -= 28
            doc.setFillColor(colors.HexColor("#75817a"))
            doc.setFont("Helvetica", 9)
            doc.drawString(56, 52, "Paper QA Agent - synthetic fixture")
            doc.drawRightString(540, 52, f"PDF page {page_number} / {pages}")
            doc.showPage()
        doc.save()
        manifest["papers"].append({"filename": filename, "title": title, "pages": pages,
                                    "source": "Generated from the project's fictional A/B/C example",
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "evidence_pdf_page": pages, "verbatim_lines": facts})
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Created three explicitly fictional teaching PDFs and SHA-256 manifest.")


if __name__ == "__main__":
    main()
