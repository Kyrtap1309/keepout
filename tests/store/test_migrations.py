import sqlite3
from pathlib import Path
from typing import Any, Final, TypedDict

import pytest
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from builders import base_revision_store
from sqlalchemy import (
    URL,
    Column,
    Engine,
    String,
    TextClause,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import IntegrityError

from keepout.errors import StoreError
from keepout.store import JobQueue, Store, db, models, open_store, upgrade
from keepout.store.db import head
from keepout.store.models import Base, JobKind

EVENT = text(
    "INSERT INTO events (id, object1, object2, tca_min, tca_max, created_at)"
    " VALUES ('0123abcd', '1', '2', 't', 't', 't')"
)
CDM = text(
    "INSERT INTO cdms (message_id, event_id, creation_date, tca, digest,"
    " content, format, origin, ingested_at)"
    " VALUES ('m', :event, 't', 't', 'd', 'c', :format, 'screen', 't')"
)


ForeignKey = tuple[
    str, tuple[str, ...], str, tuple[str, ...], tuple[tuple[str, Any], ...]
]
"""Name, columns, referred table and columns, options (``ondelete``...)."""


class TableSchema(TypedDict):
    """What SQLite reflection sees of a table."""

    columns: list[tuple[str, str, bool, str | None]]
    primary_key: list[str]
    foreign_keys: list[ForeignKey]
    indexes: list[tuple[str, tuple[str, ...], bool]]
    """Name, columns, unique."""
    unique: list[tuple[str, tuple[str, ...]]]
    checks: list[tuple[str, str]]


def _schema(engine: Engine) -> dict[str, TableSchema]:
    """Everything SQLite reflection sees of each table, comparable as text."""
    inspector = inspect(engine)
    schema: dict[str, TableSchema] = {}
    for table in inspector.get_table_names():
        if table == "alembic_version":
            continue
        columns = [
            (c["name"], str(c["type"]), c["nullable"], c["default"])
            for c in inspector.get_columns(table)
        ]
        foreign_keys = sorted(
            (
                str(fk["name"]),
                tuple(fk["constrained_columns"]),
                str(fk["referred_table"]),
                tuple(fk["referred_columns"]),
                tuple(sorted(fk.get("options", {}).items())),
            )
            for fk in inspector.get_foreign_keys(table)
        )
        schema[table] = {
            "columns": columns,
            "primary_key": inspector.get_pk_constraint(table)["constrained_columns"],
            "foreign_keys": foreign_keys,
            "indexes": sorted(
                (str(i["name"]), tuple(map(str, i["column_names"])), i["unique"])
                for i in inspector.get_indexes(table)
            ),
            "unique": sorted(
                (str(u["name"]), tuple(u["column_names"]))
                for u in inspector.get_unique_constraints(table)
            ),
            "checks": sorted(
                (str(c["name"]), " ".join(c["sqltext"].split()))
                for c in inspector.get_check_constraints(table)
            ),
        }
    return schema


def test_models_match_the_migrations(tmp_path: Path) -> None:
    # open_store builds a new file by running the migrations, not from the
    # models. test_alembic compares the two as Alembic's autogenerate does,
    # which misses CHECK constraints and ON DELETE actions: the reflected
    # schema is compared here with a file the models build.
    from_models = create_engine(URL.create("sqlite", database=str(tmp_path / "m.db")))
    Base.metadata.create_all(from_models)
    with open_store(tmp_path / "keepout.db") as store:
        assert _schema(store.engine) == _schema(from_models)
    from_models.dispose()


def test_the_schema_holds_its_rules(tmp_path: Path) -> None:
    # The comparison above is only as good as the models: the rules the plan
    # sets, checked on the migrated file.
    with open_store(tmp_path / "keepout.db") as store:
        schema = _schema(store.engine)
    on_delete = {
        (table, fk[1]): dict(fk[4]).get("ondelete")
        for table, rules in schema.items()
        for fk in rules["foreign_keys"]
    }
    assert on_delete == {
        ("cdms", ("event_id",)): "CASCADE",
        ("cdms", ("catalog_id",)): "RESTRICT",
        ("catalog_files", ("catalog_id",)): "CASCADE",
        ("results", ("message_id",)): "CASCADE",
        ("jobs", ("event_id",)): "CASCADE",
        ("jobs", ("message_id",)): "CASCADE",
    }
    assert schema["cdms"]["checks"] == [("ck_cdms_format", "format IN ('kvn', 'xml')")]
    indexes = {i[0] for rules in schema.values() for i in rules["indexes"]}
    assert {
        "ix_cdms_catalog_id",
        "ix_catalog_files_catalog_id",
        "ix_jobs_event_id_status",
        "ix_jobs_message_id",
        "ix_jobs_status_created_at",
    } <= indexes
    assert "ix_jobs_status" not in indexes
    assert "ix_jobs_event_id" not in indexes  # 0003: a prefix of event_id, status
    assert schema["jobs"]["checks"] == [
        ("ck_jobs_kind", "kind IN ('maneuver', 'refresh')"),
        (
            "ck_jobs_status",
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
        ),
    ]


def test_upgrade_reports_the_revisions(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    base_revision_store(path)
    with pytest.raises(StoreError, match="keepout store upgrade"):
        open_store(path)
    assert upgrade(path) == (None, head())
    assert upgrade(str(path)) == (head(), head())
    with open_store(path, create=False):
        pass


def test_upgrade_builds_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    path.touch()
    assert upgrade(path) == (None, head())


def test_upgrade_needs_a_store(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    with pytest.raises(StoreError, match="no store at"):
        upgrade(path)
    assert not path.exists()


def test_upgrade_refuses_a_newer_revision(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    with open_store(path):
        pass
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    with connection:
        connection.execute("UPDATE alembic_version SET version_num = 'ffff'")
    connection.close()
    before = path.read_bytes()
    with pytest.raises(StoreError, match=r"newer keepout \(revision ffff\)"):
        upgrade(path)
    assert path.read_bytes() == before


def test_upgrade_refuses_another_application_s_database(tmp_path: Path) -> None:
    path = tmp_path / "other.db"
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE notes (body TEXT)")
    connection.close()
    before = path.read_bytes()
    with pytest.raises(StoreError, match="is not a keepout store"):
        upgrade(path)
    assert path.read_bytes() == before


def test_deleting_an_event_deletes_its_messages_and_results(tmp_path: Path) -> None:
    with open_store(tmp_path / "keepout.db") as store:
        with store.write() as session:
            session.execute(EVENT)
            session.execute(CDM.bindparams(event="0123abcd", format="kvn"))
            session.execute(
                text(
                    "INSERT INTO results (message_id, key, payload, created_at)"
                    " VALUES ('m', 'k', '{}', 't')"
                )
            )
        with store.write() as session:
            session.execute(text("DELETE FROM events"))
        with store.read() as session:
            assert session.scalar(text("SELECT count(*) FROM cdms")) == 0
            assert session.scalar(text("SELECT count(*) FROM results")) == 0


@pytest.mark.parametrize(
    ("event", "cdm_format", "refusal"),
    [("nothing", "kvn", "FOREIGN KEY"), ("0123abcd", "json", "CHECK")],
)
def test_a_message_needs_its_event_and_a_known_format(
    tmp_path: Path, event: str, cdm_format: str, refusal: str
) -> None:
    with open_store(tmp_path / "keepout.db") as store:
        with store.write() as session:
            session.execute(EVENT)
        with pytest.raises(IntegrityError, match=refusal):
            _insert(store, CDM.bindparams(event=event, format=cdm_format))


def _insert(store: Store, row: TextClause) -> None:
    with store.write() as session:
        session.execute(row)


def _populated(path: Path) -> Store:
    store = open_store(path)
    with store.write() as session:
        session.execute(text("INSERT INTO catalogs (id, label) VALUES (1, 'gp')"))
        session.execute(
            text(
                "INSERT INTO catalog_files (catalog_id, name, path, sha256)"
                " VALUES (1, 'gp.csv', '/gp.csv', 'x')"
            )
        )
        session.execute(EVENT)
        session.execute(CDM.bindparams(event="0123abcd", format="kvn"))
        session.execute(text("UPDATE cdms SET catalog_id = 1"))
        session.execute(
            text(
                "INSERT INTO results (message_id, key, payload, created_at)"
                " VALUES ('m', 'k', '{}', 't')"
            )
        )
    return store


def _counts(store: Store) -> dict[str, int]:
    tables = ("events", "cdms", "results", "catalogs", "catalog_files")
    with store.read() as session:
        return {
            table: int(session.scalar(text(f"SELECT count(*) FROM {table}")) or 0)  # noqa: S608 (fixed names)
            for table in tables
        }


def test_a_catalog_of_stored_messages_cannot_be_deleted(tmp_path: Path) -> None:
    # A provenance row never deletes primary data: RESTRICT, not CASCADE.
    with _populated(tmp_path / "keepout.db") as store:
        with pytest.raises(IntegrityError, match="FOREIGN KEY"):
            _insert(store, text("DELETE FROM catalogs"))
        with store.write() as session:
            session.execute(text("UPDATE cdms SET catalog_id = NULL"))
            session.execute(text("DELETE FROM catalogs"))
        assert _counts(store) == {
            "events": 1,
            "cdms": 1,
            "results": 1,
            "catalogs": 0,
            "catalog_files": 0,
        }


def test_a_batch_alter_keeps_the_rows_of_other_tables(tmp_path: Path) -> None:
    # SQLite alters a table by copying it and dropping the old one: with
    # foreign keys enforced, the drop would cascade to the event's messages.
    path = tmp_path / "keepout.db"
    with _populated(path) as store:
        before = _counts(store)
        with db._migrating(path) as connection:
            operations = Operations(
                MigrationContext.configure(connection, opts={"render_as_batch": True})
            )
            with operations.batch_alter_table("events", recreate="always") as batch:
                batch.add_column(Column("note", String(), nullable=True))
        assert _counts(store) == before


def test_a_migration_that_breaks_a_foreign_key_is_rolled_back(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    with _populated(path) as store:
        before = _counts(store)
        with (
            pytest.raises(StoreError, match="foreign key"),
            db._migrating(path) as connection,
        ):
            connection.execute(text("DELETE FROM events"))
        assert _counts(store) == before


def _job(status: str = "queued", kind: str = "maneuver") -> TextClause:
    return text(
        "INSERT INTO jobs (event_id, message_id, key, params, status,"
        " attempt, claims, created_at, kind)"
        " VALUES ('0123abcd', 'm', 'k', '{}', :status, 0, 0, 't', :kind)"
    ).bindparams(status=status, kind=kind)


def test_a_job_id_is_never_reused(tmp_path: Path) -> None:
    # A worker's late write to a deleted job must not reach a new one.
    with open_store(tmp_path / "keepout.db") as store:
        with store.write() as session:
            session.execute(EVENT)
            session.execute(CDM.bindparams(event="0123abcd", format="kvn"))
            session.execute(_job())
            first = session.scalar(text("SELECT max(id) FROM jobs"))
            session.execute(text("DELETE FROM jobs"))
            session.execute(_job())
            second = session.scalar(text("SELECT max(id) FROM jobs"))
        assert first is not None
        assert second is not None
        assert second > first


def test_a_job_needs_a_known_status(tmp_path: Path) -> None:
    with open_store(tmp_path / "keepout.db") as store:
        with store.write() as session:
            session.execute(EVENT)
            session.execute(CDM.bindparams(event="0123abcd", format="kvn"))
        with pytest.raises(IntegrityError, match="CHECK"):
            _insert(store, _job(status="done"))


def test_a_job_needs_a_known_kind(tmp_path: Path) -> None:
    with open_store(tmp_path / "keepout.db") as store:
        with store.write() as session:
            session.execute(EVENT)
            session.execute(CDM.bindparams(event="0123abcd", format="kvn"))
        with pytest.raises(IntegrityError, match="CHECK"):
            _insert(store, _job(kind="audit"))


_SET_JOB: Final = {
    "kind": text("UPDATE jobs SET kind = :value"),
    "status": text("UPDATE jobs SET status = :value"),
}


@pytest.mark.parametrize(
    ("column", "value", "enum"),
    [("kind", "audit", "JobKind"), ("status", "done", "JobStatus")],
)
def test_a_job_of_an_unknown_kind_or_status_is_a_store_error(
    tmp_path: Path, column: str, value: str, enum: str
) -> None:
    # A store written by hand, its checks off: the job reads as a store
    # error, not a bare ValueError of the enum.
    with open_store(tmp_path / "keepout.db") as store:
        with store.write() as session:
            session.execute(EVENT)
            session.execute(CDM.bindparams(event="0123abcd", format="kvn"))
            session.execute(text("PRAGMA ignore_check_constraints = ON"))
            session.execute(
                text(
                    "INSERT INTO jobs (event_id, message_id, key, params, status,"
                    " attempt, claims, created_at, kind)"
                    " VALUES ('0123abcd', 'm', 'k', '{}', 'queued', 0, 0,"
                    " '2026-11-12T12:00:00.000000000', 'maneuver')"
                )
            )
            session.execute(_SET_JOB[column], {"value": value})
            session.execute(text("PRAGMA ignore_check_constraints = OFF"))
        with pytest.raises(StoreError, match=f"^unknown {enum} '{value}' in "):
            JobQueue(store).jobs()


def test_upgrade_0002_to_0003_keeps_jobs(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            command.upgrade(db._config(connection), "0002")
            connection.execute(EVENT)
            connection.execute(CDM.bindparams(event="0123abcd", format="kvn"))
            connection.execute(
                text(
                    "INSERT INTO jobs (event_id, message_id, key, params, status,"
                    " attempt, claims, created_at)"
                    " VALUES ('0123abcd', 'm', 'k', '{}', 'queued', 0, 0,"
                    " '2026-11-12T12:00:00.000000000')"
                )
            )
    finally:
        engine.dispose()
    assert upgrade(path) == ("0002", head())
    with open_store(path, create=False) as store:
        with store.read() as session:
            kind = session.scalar(text("SELECT kind FROM jobs"))
            row = session.get_one(models.Job, 1)
        assert kind == "maneuver"
        assert row.kind is JobKind.MANEUVER
        assert row.status == "queued"


def test_the_dashboard_indexes(tmp_path: Path) -> None:
    with open_store(tmp_path / "keepout.db") as store:
        schema = _schema(store.engine)
    indexes = {
        name: (columns, unique)
        for rules in schema.values()
        for name, columns, unique in rules["indexes"]
    }
    assert indexes["ix_events_tca_max"] == (("tca_max",), False)
    assert indexes["ix_jobs_event_id_status"] == (("event_id", "status"), False)
    # Those the first revisions made already serve the dashboard's queries.
    assert indexes["ix_events_object1_object2_tca_min"][0][:2] == ("object1", "object2")
    assert indexes["ix_cdms_event_id_creation_date"][0] == ("event_id", "creation_date")
    unique = schema["results"]["unique"]
    assert ("uq_results_message_id_key", ("message_id", "key")) in unique
