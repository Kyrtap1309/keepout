"""Alembic environment of the conjunction store.

The migrations run online, on the connection :mod:`keepout.store.db` puts in
the configuration's ``connection`` attribute, inside its transaction.
"""

from alembic import context
from sqlalchemy import Connection

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


def _connection() -> Connection:
    """The connection keepout gives the migrations.

    Raises
    ------
    StoreError
        If there is none (the ``alembic`` command, or offline mode).
    """
    connection = context.config.attributes.get("connection")
    if context.is_offline_mode() or not isinstance(connection, Connection):
        raise StoreError("the store's migrations run through keepout store upgrade")
    return connection


run_migrations(_connection())
