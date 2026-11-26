import os
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import TextClause, text

from keepout.errors import StoreError
from keepout.store import Store, db, open_store, upgrade
from keepout.store.db import head


def _revision(path: Path) -> str | None:
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    connection.close()
    return None if row is None else str(row[0])


def _set_revision(path: Path, revision: str | None) -> None:
    # Also out of WAL mode: a refused file keeping its bytes then shows that
    # opening it did not switch it to WAL either.
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    with connection:
        connection.execute("DELETE FROM alembic_version")
        if revision is not None:
            connection.execute(
                "INSERT INTO alembic_version (version_num) VALUES (?)", (revision,)
            )
    connection.close()


def _created(path: Path) -> Path:
    with open_store(path):
        pass
    return path


def test_a_new_file_is_created_at_head(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "keepout.db"
    with open_store(path) as store:
        assert isinstance(store, Store)
        assert store.path == path
        with store.read() as session:
            tables = set(
                session.scalars(
                    text("SELECT name FROM sqlite_master WHERE type = 'table'")
                )
            )
    assert {"events", "cdms", "catalogs", "catalog_files", "results"} <= tables
    assert _revision(path) == head()


def test_an_old_sqlite_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.34.1")
    path = tmp_path / "keepout.db"
    with pytest.raises(
        StoreError, match=r"^the store needs SQLite 3\.35 or newer, found 3\.34\.1$"
    ):
        open_store(path)
    assert not path.exists()


def test_a_path_may_be_a_string(tmp_path: Path) -> None:
    with open_store(str(tmp_path / "keepout.db")) as store:
        assert store.path == tmp_path / "keepout.db"


def test_a_path_may_hold_url_characters(tmp_path: Path) -> None:
    path = tmp_path / "a?b#c.db"
    with open_store(path) as store:
        assert store.path == path
    assert path.is_file()
    assert _revision(path) == head()


def _empty(path: Path) -> None:
    path.touch()


def _tableless(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 7")  # a header, no table
    connection.close()


@pytest.mark.parametrize("make", [_empty, _tableless], ids=["empty", "tableless"])
def test_an_empty_database_is_created_at_head(
    tmp_path: Path, make: Callable[[Path], None]
) -> None:
    path = tmp_path / "keepout.db"
    make(path)
    with open_store(path):
        pass
    assert _revision(path) == head()


def test_an_empty_file_is_no_store_without_create(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    path.touch()
    with pytest.raises(StoreError, match="no store at"):
        open_store(path, create=False)
    assert path.read_bytes() == b""


def test_another_application_s_database_is_refused_and_left_untouched(
    tmp_path: Path,
) -> None:
    path = tmp_path / "other.db"
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        connection.execute("INSERT INTO notes (body) VALUES ('keep me')")
    connection.close()
    before = path.read_bytes()
    with pytest.raises(StoreError, match=r"other\.db is not a keepout store"):
        open_store(path)
    assert path.read_bytes() == before


def test_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match=re.escape(str(tmp_path))):
        open_store(tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
def test_an_unwritable_directory_is_refused(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(StoreError, match=r"keepout\.db"):
            open_store(locked / "keepout.db")
        with pytest.raises(StoreError, match=r"keepout\.db"):
            open_store(locked / "inner" / "keepout.db")
    finally:
        locked.chmod(0o700)


def test_a_missing_file_is_not_created_on_request(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    with pytest.raises(StoreError, match="no store at"):
        open_store(path, create=False)
    assert not path.exists()


def test_an_existing_store_opens_without_create(tmp_path: Path) -> None:
    path = _created(tmp_path / "keepout.db")
    with open_store(path, create=False) as store:
        assert store.path == path


def test_an_older_revision_is_refused_and_left_untouched(tmp_path: Path) -> None:
    path = _created(tmp_path / "keepout.db")
    _set_revision(path, None)  # alembic stamp base
    before = path.read_bytes()
    with pytest.raises(StoreError, match="older keepout: run keepout store upgrade"):
        open_store(path)
    assert path.read_bytes() == before
    assert _revision(path) is None


def test_a_newer_revision_is_refused(tmp_path: Path) -> None:
    path = _created(tmp_path / "keepout.db")
    _set_revision(path, "ffffffffffff")
    before = path.read_bytes()
    with pytest.raises(StoreError, match=r"newer keepout \(revision ffffffffffff\)"):
        open_store(path)
    assert path.read_bytes() == before


def test_a_file_that_is_not_a_database_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not a database, just some longer text " * 10)
    with pytest.raises(StoreError, match="not a keepout store"):
        open_store(path)


def test_connections_use_wal_and_foreign_keys(tmp_path: Path) -> None:
    with open_store(tmp_path / "keepout.db") as store, store.read() as session:
        assert session.scalar(text("PRAGMA journal_mode")) == "wal"
        assert session.scalar(text("PRAGMA foreign_keys")) == 1
        assert session.scalar(text("PRAGMA busy_timeout")) == 30_000


def _journal_mode(path: Path) -> str:
    connection = sqlite3.connect(path)
    mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
    connection.close()
    return mode


def test_a_new_store_is_in_wal_mode_before_its_first_use(tmp_path: Path) -> None:
    # Switching to WAL takes an exclusive lock: left to the first connection
    # of each process, two processes switching at once fail "locked".
    path = tmp_path / "keepout.db"
    open_store(path).close()
    assert _journal_mode(path) == "wal"


def test_opening_a_store_out_of_wal_mode_switches_it(tmp_path: Path) -> None:
    path = _created(tmp_path / "keepout.db")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.close()
    open_store(path).close()
    assert _journal_mode(path) == "wal"


def _locked_for_writing(path: Path) -> bool:
    other = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        other.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as error:
        locked = "locked" in str(error)
        if not locked:
            raise
        return locked
    else:
        other.execute("ROLLBACK")
        return False
    finally:
        other.close()


def test_a_write_session_takes_the_write_lock_at_once(tmp_path: Path) -> None:
    # BEGIN IMMEDIATE on entry: a second writer waits at its start instead of
    # failing to upgrade a read lock halfway through.
    with open_store(tmp_path / "keepout.db") as store:
        with store.write():
            assert _locked_for_writing(store.path)
        assert not _locked_for_writing(store.path)


def test_a_read_session_does_not_block_writers(tmp_path: Path) -> None:
    with open_store(tmp_path / "keepout.db") as store, store.read() as session:
        session.scalar(text("SELECT count(*) FROM events"))
        assert not _locked_for_writing(store.path)


def _insert_and_fail(store: Store, insert: TextClause) -> None:
    with store.write() as session:
        session.execute(insert, {"label": "dropped"})
        raise RuntimeError("rolled back")


def test_a_write_session_commits_or_rolls_back(tmp_path: Path) -> None:
    insert = text("INSERT INTO catalogs (label) VALUES (:label)")
    with open_store(tmp_path / "keepout.db") as store:
        with store.write() as session:
            session.execute(insert, {"label": "kept"})
        with pytest.raises(RuntimeError, match="rolled back"):
            _insert_and_fail(store, insert)
        with store.read() as session:
            labels = list(session.scalars(text("SELECT label FROM catalogs")))
    assert labels == ["kept"]


def test_closing_a_store_twice_is_harmless(tmp_path: Path) -> None:
    store = open_store(tmp_path / "keepout.db")
    store.close()
    store.close()


@pytest.mark.parametrize("call", [open_store, upgrade])
def test_a_path_that_cannot_be_checked_is_a_store_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: Callable[[Path], object]
) -> None:
    def unsearchable(self: Path) -> bool:
        raise PermissionError(13, "Permission denied", str(self))

    path = tmp_path / "locked" / "keepout.db"
    monkeypatch.setattr(Path, "exists", unsearchable)
    expected = f"cannot use the store {re.escape(str(path))}: .*denied"
    with pytest.raises(StoreError, match=expected):
        call(path)


def test_a_store_made_meanwhile_by_a_newer_keepout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another process made the file between the check and the write lock.
    path = _created(tmp_path / "keepout.db")
    _set_revision(path, "ffffffffffff")
    monkeypatch.setattr(db, "_read_schema", lambda _: db._Schema(True, None))
    with pytest.raises(StoreError, match="newer keepout"):
        open_store(path)
    monkeypatch.setattr(db, "_read_schema", lambda _: db._Schema(False, None))
    with pytest.raises(StoreError, match="newer keepout"):
        upgrade(path)
    assert _revision(path) == "ffffffffffff"


def test_a_store_made_meanwhile_at_head_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _created(tmp_path / "keepout.db")
    monkeypatch.setattr(db, "_read_schema", lambda _: db._Schema(True, None))
    with open_store(path) as store:
        assert store.path == path
