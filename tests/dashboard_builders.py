"""Helpers of the dashboard's tests: the app's script, how long a run of it
may take, the type of the ``app_test`` fixture, and a store taken down to an
older schema."""

from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol

from alembic import command
from sqlalchemy import create_engine
from streamlit.testing.v1 import AppTest

from keepout.store import db as store_db

APP: Final = Path(store_db.__file__).parents[1] / "dashboard" / "app.py"
"""The dashboard's entry script."""

DEFAULT_PAGE: Final = "events"
"""The page the app opens on."""

TIMEOUT: Final = 30.0
"""Seconds a run of the app, or of a widget's script, may take: the first
imports the numerical libraries."""


class OpenApp(Protocol):
    """The ``app_test`` fixture: the app run on a page."""

    def __call__(
        self, page: str = DEFAULT_PAGE, *, query: Mapping[str, str] | None = None
    ) -> AppTest:
        """The app run on ``page`` (``events``, ``event`` or ``planner``, the
        scripts in ``keepout/dashboard/views``), with the query parameters
        ``query``."""
        ...


def open_app(
    page: str = DEFAULT_PAGE, *, query: Mapping[str, str] | None = None
) -> AppTest:
    """The app run on ``page`` with the query parameters ``query``, on the
    store and settings the environment names (see ``app_test``)."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.query_params.update(query or {})
    app.run()
    if page != DEFAULT_PAGE:
        # A page of st.navigation is known once the app has run.
        app.switch_page(f"views/{page}.py")
        app.query_params.update(query or {})
        app.run()
    return app


def downgrade_store(path: Path, revision: str) -> None:
    """Take the store at ``path`` down to ``revision`` of its schema, as an
    older keepout left it."""
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            command.downgrade(store_db._config(connection), revision)
    finally:
        engine.dispose()
