from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from app import config as configuration
from app.config import ROOT
from sqlalchemy import create_engine, inspect, text


def test_empty_database_upgrades_entire_chain_without_duplicate_columns(tmp_path, settings, monkeypatch):
    settings.database_url = f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    monkeypatch.setattr(configuration, "get_settings", lambda: settings)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend/migrations"))
    heads = ScriptDirectory.from_config(config).get_heads()
    assert len(heads) == 1
    command.upgrade(config, "head")
    engine = create_engine(settings.database_url)
    try:
        schema = inspect(engine)
        assert {"category", "answer_sha256", "idempotency_key"} <= {
            c["name"] for c in schema.get_columns("review_records")}
        assert {"test_environment", "verification_sha256", "workspace_state"} <= {
            c["name"] for c in schema.get_columns("code_repair_tasks")}
        assert {"answer_approval_records", "accepted_answer_experiences"} <= set(schema.get_table_names())
        assert "approval_expires_at" in {c["name"] for c in schema.get_columns("external_operations")}
        assert {"run_id", "answer_id", "reviewer_id", "decision", "answer_sha256", "reason"} <= {
            c["name"] for c in schema.get_columns("answer_approval_records")}
        assert {"run_id", "answer_id", "corpus_id", "reviewer_id", "answer_sha256", "question"} <= {
            c["name"] for c in schema.get_columns("accepted_answer_experiences")}
        approval_unique = {tuple(item["column_names"]) for item in
                           schema.get_unique_constraints("answer_approval_records")}
        experience_unique = {tuple(item["column_names"]) for item in
                             schema.get_unique_constraints("accepted_answer_experiences")}
        assert {("run_id",), ("answer_id",)} <= approval_unique
        assert {("run_id",)} <= experience_unique
        approval_foreign_keys = {item["referred_table"] for item in
                                 schema.get_foreign_keys("answer_approval_records")}
        experience_foreign_keys = {item["referred_table"] for item in
                                   schema.get_foreign_keys("accepted_answer_experiences")}
        assert {"question_runs", "answer_revisions", "users"} <= approval_foreign_keys
        assert {"question_runs", "answer_revisions", "corpora", "users"} <= experience_foreign_keys
        with engine.connect() as db:
            assert db.execute(text("select version_num from alembic_version")).scalar() == heads[0]
    finally:
        engine.dispose()


def test_code_evidence_migration_adds_missing_fields_and_preserves_old_rows(tmp_path):
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend/migrations"))
    migration = ScriptDirectory.from_config(config).get_revision("0007").module
    engine = create_engine(f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}")
    try:
        with engine.begin() as db:
            db.execute(text("create table code_repair_tasks (id text primary key, status text not null)"))
            db.execute(text("create table code_repair_reviews (id text primary key, reason text)"))
            db.execute(text("insert into code_repair_tasks values ('task', 'rejected')"))
            db.execute(text("insert into code_repair_reviews values ('review', 'original feedback')"))
            with Operations.context(MigrationContext.configure(db)):
                migration.upgrade()
            assert db.execute(text("select workspace_state from code_repair_tasks")).scalar() == "restored"
            assert db.execute(text("select reason from code_repair_reviews")).scalar() == "original feedback"
            assert db.execute(text("select feedback_category from code_repair_reviews")).scalar() == "unclear"
    finally:
        engine.dispose()


def test_external_approval_expiry_migration_downgrades_without_losing_operation_rows(tmp_path):
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend/migrations"))
    migration = ScriptDirectory.from_config(config).get_revision("0010").module
    engine = create_engine(f"sqlite:///{(tmp_path / 'approval-expiry.db').as_posix()}")
    try:
        with engine.begin() as db:
            db.execute(text("CREATE TABLE external_operations (id TEXT PRIMARY KEY, status TEXT NOT NULL)"))
            db.execute(text("INSERT INTO external_operations VALUES ('operation-1', 'approved')"))
            with Operations.context(MigrationContext.configure(db)):
                migration.upgrade()
            assert "approval_expires_at" in {column["name"] for column in inspect(db).get_columns(
                "external_operations")}

        with engine.begin() as db:
            with Operations.context(MigrationContext.configure(db)):
                migration.downgrade()
            assert "approval_expires_at" not in {column["name"] for column in inspect(db).get_columns(
                "external_operations")}
            assert db.execute(text("SELECT status FROM external_operations WHERE id='operation-1'")).scalar() == "approved"
    finally:
        engine.dispose()
