import sqlite3
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
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
from keepout.store import Store, db, open_store, upgrade
from keepout.store.db import HEAD
from keepout.store.models import Base

EVENT = text(
    "INSERT INTO events (id, object1, object2, tca_min, tca_max, created_at)"
    " VALUES ('0123abcd', '1', '2', 't', 't', 't')"
)
CDM = text(
    "INSERT INTO cdms (message_id, event_id, creation_date, tca, digest,"
    " content, format, origin, ingested_at)"
    " VALUES ('m', :event, 't', 't', 'd', 'c', :format, 'screen', 't')"
)


def _schema(engine: Engine) -> dict[str, dict[str, object]]:
    """Everything SQLite reflection sees of each table, comparable as text."""
    inspector = inspect(engine)
    schema: dict[str, dict[str, object]] = {}
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
    # models: any difference between the two shows here. compare_metadata
    # misses CHECK constraints and ON DELETE actions, so the reflected
    # schema is also compared with a file the models build.
    from_models = create_engine(URL.create("sqlite", database=str(tmp_path / "m.db")))
    Base.metadata.create_all(from_models)
    with open_store(tmp_path / "keepout.db") as store:
        with store.engine.connect() as conn:
            context = MigrationContext.configure(
                conn, opts={"compare_type": True, "compare_server_default": True}
            )
            assert compare_metadata(context, Base.metadata) == []
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
        for fk in rules["foreign_keys"]  # type: ignore[attr-defined]
    }
    assert on_delete == {
        ("cdms", ("event_id",)): "CASCADE",
        ("cdms", ("catalog_id",)): "RESTRICT",
        ("catalog_files", ("catalog_id",)): "CASCADE",
        ("results", ("message_id",)): "CASCADE",
    }
    assert schema["cdms"]["checks"] == [("ck_cdms_format", "format IN ('kvn', 'xml')")]
    assert ("archived", "BOOLEAN", False, "0") in schema["events"]["columns"]  # type: ignore[operator]
    indexes = {i[0] for rules in schema.values() for i in rules["indexes"]}  # type: ignore[attr-defined]
    assert {"ix_cdms_catalog_id", "ix_catalog_files_catalog_id"} <= indexes


def _alembic_version_only(path: Path) -> None:
    """A file at the base revision: Alembic's table, empty, and nothing else."""
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL,"
            " CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
        )
    connection.close()


def test_upgrade_reports_the_revisions(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    _alembic_version_only(path)
    with pytest.raises(StoreError, match="keepout store upgrade"):
        open_store(path)
    assert upgrade(path) == (None, HEAD)
    assert upgrade(str(path)) == (HEAD, HEAD)
    with open_store(path, create=False):
        pass


def test_upgrade_builds_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    path.touch()
    assert upgrade(path) == (None, HEAD)


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
        with store.read() as session:
            assert session.scalar(text("SELECT archived FROM events")) == 0
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
