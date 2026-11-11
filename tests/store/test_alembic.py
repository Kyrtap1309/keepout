"""pytest-alembic's checks of the store's migrations, on a SQLite file as
keepout migrates it: one head; every revision upgrades from the base, and
downgrades and upgrades again step by step (the stairway); the head builds
what the models describe."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from pytest_alembic.tests import (  # noqa: F401 (collected by pytest)
    test_model_definitions_match_ddl,
    test_single_head_revision,
    test_up_down_consistency,
    test_upgrade,
)
from sqlalchemy import Engine

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
