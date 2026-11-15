"""Opening a conjunction store: the SQLite engine, the schema revision and
its upgrade.

The schema is versioned by Alembic (``keepout/store/migrations``). A store
opens only at the revision this keepout was built for: an older file is
upgraded on request (:func:`upgrade`, ``keepout store upgrade``); a newer one,
or another application's database, is never touched, not even switched to
write-ahead logging.

:func:`open_store` opens a store (re-exported as
:func:`keepout.store.open_store`), checking its revision against
:data:`HEAD`. :func:`engine_for` makes the SQLAlchemy engine a
:class:`~keepout.store.Store` runs on: write-ahead logging, foreign keys, and
a wait of :data:`BUSY_TIMEOUT_MS` for another process's lock.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import URL, create_engine, event, inspect
from sqlalchemy.exc import DatabaseError

from keepout.errors import StoreError
from keepout.store.repo import WRITE, Store

if TYPE_CHECKING:
    from collections.abc import Generator
    from sqlite3 import Connection as SqliteConnection

    from sqlalchemy import Connection, Engine
    from sqlalchemy.pool import ConnectionPoolEntry

__all__ = [
    "BUSY_TIMEOUT_MS",
    "HEAD",
    "engine_for",
    "open_store",
    "upgrade",
]

BUSY_TIMEOUT_MS: Final = 30_000
"""How long a connection waits for a lock another one holds, in milliseconds
(30 s); past it, the store raises :class:`~keepout.errors.StoreError`.

SQLite's ``busy_timeout``, set on every connection of :func:`engine_for`: a
writer waits this long for another process's write to end."""

_MIGRATIONS: Final = "keepout.store:migrations"


def _config(connection: Connection | None = None) -> Config:
    """The Alembic configuration of the store, built in code (no ``alembic.ini``)."""
    config = Config()
    config.set_main_option("script_location", _MIGRATIONS)
    config.attributes["connection"] = connection
    return config


_SCRIPTS: Final = ScriptDirectory.from_config(_config())

# The package ships at least one migration, so there is a head.
HEAD: Final = _SCRIPTS.get_current_head() or ""
"""The schema revision this keepout reads and writes: the Alembic head of its
migrations. :func:`open_store` opens only a store at this revision;
:func:`upgrade` brings an older one to it."""

_KNOWN: Final = frozenset(script.revision for script in _SCRIPTS.walk_revisions())
"""Every revision of this keepout's migrations."""


def _engine(path: Path, *, wal: bool, foreign_keys: bool) -> Engine:
    """An engine on a SQLite file whose transactions keepout begins itself.

    Every connection waits up to :data:`BUSY_TIMEOUT_MS` for a lock, and
    enforces foreign keys or not; with ``wal``, it also switches the file to
    write-ahead logging (a lasting change to the file). Transactions start
    with ``BEGIN``, or ``BEGIN IMMEDIATE`` with the
    :data:`~keepout.store.repo.WRITE` execution option.
    """
    engine = create_engine(URL.create("sqlite", database=str(path)))

    @event.listens_for(engine, "connect")
    def _connect(connection: SqliteConnection, _: ConnectionPoolEntry) -> None:
        # The begin listener below emits BEGIN, not the sqlite3 module; the
        # pragmas run outside any transaction (inside one, foreign_keys is
        # ignored).
        connection.isolation_level = None
        cursor = connection.cursor()
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        cursor.execute(f"PRAGMA foreign_keys={'ON' if foreign_keys else 'OFF'}")
        if wal:
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    @event.listens_for(engine, "begin")
    def _begin(connection: Connection) -> None:
        # The writers set the option to True (execution options are Any).
        immediate = connection.get_execution_options().get(WRITE) is True
        connection.exec_driver_sql("BEGIN IMMEDIATE" if immediate else "BEGIN")

    return engine


def engine_for(path: Path) -> Engine:
    """A SQLAlchemy engine on a SQLite file, set up for the store.

    Every connection uses write-ahead logging (readers and a writer do not
    block each other), waits up to :data:`BUSY_TIMEOUT_MS` for a lock and
    enforces foreign keys. Transactions start with ``BEGIN``, or with
    ``BEGIN IMMEDIATE`` on a connection with the
    :data:`~keepout.store.repo.WRITE` execution option, so two writers queue
    instead of one failing to upgrade its read lock.

    Switching to write-ahead logging changes the file: :func:`open_store`
    makes this engine only once it has checked the file's revision.

    Parameters
    ----------
    path
        The database file; SQLite creates it on first connection.

    Returns
    -------
    Engine
        The engine; :meth:`~sqlalchemy.engine.Engine.dispose` closes its
        connections.
    """
    return _engine(path, wal=True, foreign_keys=True)


@dataclass(frozen=True)
class _Schema:
    """What a database holds of keepout's schema."""

    empty: bool
    """No table at all: a new file (SQLite makes a missing file empty)."""
    revision: str | None
    """Alembic's revision; ``None`` for an empty file or the base revision."""


def _store_message(path: str | Path | None, error: DatabaseError) -> str:
    """The store's message of a database error, naming the file."""
    if "not a database" in str(error.orig):
        return f"{path} is not a keepout store: {error.orig}"
    return f"cannot use the store {path}: {error.orig}"


@contextmanager
def _connected(engine: Engine, *, write: bool) -> Generator[Connection]:
    """A connection in a transaction, a write one (``BEGIN IMMEDIATE``) or not.

    Raises
    ------
    StoreError
        On a database error: the file cannot be opened (a directory, an
        unwritable place), is not a SQLite database, or stayed locked past
        the busy timeout.
    """
    if write:
        engine = engine.execution_options(**{WRITE: True})
    try:
        with engine.begin() as connection:
            yield connection
    except DatabaseError as error:
        raise StoreError(_store_message(engine.url.database, error)) from error


def _schema(connection: Connection, path: str | Path | None) -> _Schema:
    """What the connection's database holds of keepout's schema.

    Raises
    ------
    StoreError
        If it has tables but no Alembic revision table: another
        application's database.
    """
    tables = set(inspect(connection).get_table_names())
    if not tables:
        return _Schema(empty=True, revision=None)
    if "alembic_version" not in tables:
        raise StoreError(f"{path} is not a keepout store")
    revision = MigrationContext.configure(connection).get_current_revision()
    return _Schema(empty=False, revision=revision)


def _require_known(path: str | Path | None, schema: _Schema) -> None:
    """Check that this keepout knows the database's revision.

    Raises
    ------
    StoreError
        If it does not: a newer keepout made the database.
    """
    if schema.revision is not None and schema.revision not in _KNOWN:
        raise StoreError(
            f"{path} was made by a newer keepout (revision {schema.revision}): "
            "upgrade keepout to open it"
        )


def _require_head(path: str | Path | None, schema: _Schema) -> None:
    """Check that the database is at :data:`HEAD`, the revision this keepout
    opens.

    Raises
    ------
    StoreError
        If it is at a newer or an older revision.
    """
    if schema.revision == HEAD:
        return
    _require_known(path, schema)
    raise StoreError(f"{path} was made by an older keepout: run keepout store upgrade")


@contextmanager
def _inspected(path: Path) -> Generator[Engine]:
    """An engine that changes nothing on connection (no write-ahead logging),
    disposed of on exit."""
    engine = _engine(path, wal=False, foreign_keys=True)
    try:
        yield engine
    finally:
        engine.dispose()


def _read_schema(path: Path) -> _Schema:
    """What a file holds of keepout's schema, read without changing it.

    Raises
    ------
    StoreError
        If the file cannot be read, is not a SQLite database or is another
        application's database.
    """
    with _inspected(path) as engine, _connected(engine, write=False) as connection:
        return _schema(connection, path)


@contextmanager
def _migrating(path: Path) -> Generator[Connection]:
    """A connection to migrate a store on, in a write transaction.

    Foreign keys are off, set before the transaction begins: SQLite alters
    a table by copying it and dropping the old one, and with foreign keys
    on the drop would delete, by ``ON DELETE CASCADE``, the rows of other
    tables that refer to it. They are checked before the commit instead.

    Raises
    ------
    StoreError
        If the migrations leave a row whose foreign key refers to nothing
        (the transaction is rolled back), or on a database error.
    """
    engine = _engine(path, wal=False, foreign_keys=False)
    try:
        with _connected(engine, write=True) as connection:
            yield connection
            broken = connection.exec_driver_sql("PRAGMA foreign_key_check").all()
            if broken:
                # A raw PRAGMA's rows are untyped: the first column is the table.
                tables = sorted({str(row[0]) for row in broken})  # pyright: ignore[reportAny]
                raise StoreError(
                    f"migrating {path} broke foreign keys in {', '.join(tables)}"
                )
    finally:
        engine.dispose()


def _to_head(connection: Connection) -> None:
    """Run the migrations from the connection's revision to :data:`HEAD`."""
    command.upgrade(_config(connection), "head")


def _create(path: Path) -> None:
    """Build the schema in an empty file (made with its directory if missing).

    Raises
    ------
    StoreError
        If the directory cannot be made, the file cannot be written, or
        another process made the file meanwhile at a revision this keepout
        cannot open.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise StoreError(f"cannot create the store {path}: {error}") from error
    with _migrating(path) as connection:
        # Under the write lock: another process may have made the file
        # meanwhile.
        schema = _schema(connection, path)
        if schema.empty:
            _to_head(connection)
        else:
            _require_head(path, schema)


def _no_store(path: Path) -> str:
    """The error for a store that is not there: how to make one."""
    return (
        f"no store at {path}: make one by adding CDMs to it, "
        f"keepout store add CDM... --store {path}"
    )


def _exists(path: Path) -> bool:
    """Whether there is a file (or anything) at ``path``.

    Raises
    ------
    StoreError
        If that cannot be told, e.g. its directory cannot be searched.
    """
    try:
        return path.exists()
    except OSError as error:
        raise StoreError(f"cannot use the store {path}: {error}") from error


_SQLITE: Final = (3, 35)
"""The oldest SQLite the store runs on: the first with ``UPDATE … RETURNING``."""


def _require_sqlite() -> None:
    """A check that Python's SQLite is :data:`_SQLITE` or newer.

    Raises
    ------
    StoreError
        If it is older; the message names its version.
    """
    if sqlite3.sqlite_version_info < _SQLITE:
        oldest = ".".join(map(str, _SQLITE))
        raise StoreError(
            f"the store needs SQLite {oldest} or newer, found {sqlite3.sqlite_version}"
        )


def open_store(path: str | Path, *, create: bool = True) -> Store:
    """Open a conjunction store.

    Parameters
    ----------
    path
        The database file.
    create
        Whether a missing or empty file (an SQLite database without tables)
        is made a store, at the schema of this keepout, with its directory.

    Returns
    -------
    Store
        The open store; close it, or use it as a context manager.

    Raises
    ------
    StoreError
        If the file is missing or empty and ``create`` is false; if it
        cannot be opened or created (a directory, an unwritable place, a
        directory that cannot be searched, a lock held past the busy
        timeout); if it is not a SQLite database or
        is another application's database; or if it was made by an older
        keepout (``keepout store upgrade`` upgrades it) or a newer one; or
        if Python's SQLite is older than 3.35. A refused file is not changed.

    Examples
    --------
    ```python
    from pathlib import Path

    from keepout.settings import Settings
    from keepout.store import CdmInput, open_store

    with open_store("keepout.db") as store:  # created if missing
        # A sample CDM in the repository.
        report = store.ingest([CdmInput.read(Path("tests/cdm/data/tracss_v2.kvn"))])
        print(report.added, report.new_events)
        # ('TRACSS-TEST-0001',) ('5ede3a14c345',)
        settings = Settings(combined_hard_body_radius=20.0)  # [m]
        for event, assessment in store.assess(report.events, settings):
            action = assessment.decision.action
            print(event.event_id, assessment.level.key, action.label)
        # 5ede3a14c345 red maneuver
    ```
    """
    _require_sqlite()
    path = Path(path)
    schema = _read_schema(path) if _exists(path) else _Schema(empty=True, revision=None)
    if schema.empty:
        if not create:
            raise StoreError(_no_store(path))
        _create(path)
    else:
        _require_head(path, schema)
    return Store(path, engine_for(path))


def upgrade(path: str | Path) -> tuple[str | None, str]:
    """Upgrade a conjunction store to the schema of this keepout.

    Parameters
    ----------
    path
        The database file; an empty one (no tables) gets the whole schema.

    Returns
    -------
    tuple[str | None, str]
        The revision before (``None`` for a file without a schema) and after
        (:data:`HEAD`); the same when there was nothing to do.

    Raises
    ------
    StoreError
        If there is no file; it cannot be found (a directory that cannot be
        searched), opened or written, is not a SQLite
        database or is another application's database; it was made by a
        newer keepout; or a migration broke a foreign key. The file is not
        changed then.
    """
    path = Path(path)
    if not _exists(path):
        raise StoreError(_no_store(path))
    schema = _read_schema(path)
    _require_known(path, schema)
    if schema.revision == HEAD:
        return HEAD, HEAD
    with _migrating(path) as connection:
        # Again under the write lock: another process may have upgraded it.
        schema = _schema(connection, path)
        _require_known(path, schema)
        _to_head(connection)
    return schema.revision, HEAD
