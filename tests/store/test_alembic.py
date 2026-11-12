"""pytest-alembic's checks of the store's migrations, on a SQLite file as
keepout migrates it: one head; every revision upgrades from the base, and
downgrades and upgrades again step by step (the stairway); the head builds
what the models describe; and a store of the first revision keeps its rows."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from pytest_alembic import MigrationContext
from pytest_alembic.tests import (  # noqa: F401 (collected by pytest)
    test_model_definitions_match_ddl,
    test_single_head_revision,
    test_up_down_consistency,
    test_upgrade,
)
from sqlalchemy import Engine, text

from keepout.store import db


@pytest.fixture
def alembic_config() -> Config:
    """The store's Alembic configuration, built in code as keepout builds it."""
    return db._config()


@pytest.fixture
def alembic_engine(tmp_path: Path) -> Iterator[Engine]:
    """An engine on a new file, as keepout migrates one: foreign keys off,
    no write-ahead logging."""
    engine = db._engine(tmp_path / "keepout.db", wal=False, foreign_keys=False)
    yield engine
    engine.dispose()


def test_a_populated_first_revision_store_upgrades_with_its_rows(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    alembic_runner.migrate_up_to("0001")
    alembic_runner.insert_into(
        "events",
        {
            "id": "0123abcd",
            "object1": "1",
            "object2": "2",
            "tca_min": "t",
            "tca_max": "t",
            "created_at": "t",
        },
    )
    alembic_runner.insert_into(
        "cdms",
        {
            "message_id": "m",
            "event_id": "0123abcd",
            "creation_date": "t",
            "tca": "t",
            "digest": "d",
            "content": "c",
            "format": "kvn",
            "origin": "screen",
            "ingested_at": "t",
        },
    )
    alembic_runner.insert_into(
        "results", {"message_id": "m", "key": "k", "payload": "{}", "created_at": "t"}
    )
    alembic_runner.migrate_up_to("heads")
    with alembic_engine.connect() as connection:
        rows = {
            table: connection.execute(text(f"SELECT count(*) FROM {table}")).scalar()  # noqa: S608 (fixed names)
            for table in ("events", "cdms", "results", "jobs")
        }
    assert rows == {"events": 1, "cdms": 1, "results": 1, "jobs": 0}
