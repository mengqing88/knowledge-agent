from app.core.errors import BudgetExhausted
from app.knowledge.ingestion import extract_document


def test_document_parser_checks_task_deadline_between_markdown_lines(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\nFirst section\nSecond section\n", encoding="utf-8")
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise BudgetExhausted("ingestion_time")

    try:
        extract_document(path, path.name, guard)
    except BudgetExhausted as exc:
        assert exc.category == "ingestion_time"
    else:
        raise AssertionError("ingestion deadline was not checked inside the parser loop")
