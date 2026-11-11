"""Alembic environment of the conjunction store.

The migrations run online, on the connection :mod:`keepout.store.db` puts in
the configuration's ``connection`` attribute, inside its transaction; or on a
connection of the engine put there instead, in a transaction of their own (as
pytest-alembic does).
"""

from alembic import context
from sqlalchemy import Connection, Engine

from keepout.errors import StoreError
from keepout.store.models import Base


def run_migrations(connection: Connection) -> None:
    """Run the migrations on a connection, in batch mode (SQLite alters a
    table by copying it)."""
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _connectable() -> Connection | Engine:
    """The connection, or engine, the migrations are given.

    Raises
    ------
    StoreError
        If there is none (the ``alembic`` command, or offline mode).
    """
    connectable = context.config.attributes.get("connection")
    if context.is_offline_mode() or not isinstance(connectable, Connection | Engine):
        raise StoreError("the store's migrations run through keepout store upgrade")
    return connectable


_given = _connectable()
if isinstance(_given, Engine):
    with _given.connect() as _connection:
        run_migrations(_connection)
else:
    run_migrations(_given)
