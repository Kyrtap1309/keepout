"""Helpers of the dashboard's tests: the app's script, how long a run of it
may take, the type of the ``app_test`` fixture, the environment the app is
run in, what a page shows, and a store taken down to an older schema."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final, Protocol

import pytest
from alembic import command
from astropy.time import Time
from sqlalchemy import create_engine
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, Dataframe, Node

from keepout._environment import AS_OF_VARIABLE, SETTINGS_VARIABLE, STORE_VARIABLE
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


def use_store(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    settings: Path | None = None,
    as_of: Time | None = None,
) -> None:
    """Set the variables ``keepout dashboard`` passes for the test: the
    store at ``path``, the settings file ``settings`` (``None``: unset, the
    defaults) and the time ``as_of`` (``None``: unset, the clock)."""
    monkeypatch.setenv(STORE_VARIABLE, str(path))
    for variable, value in (
        (SETTINGS_VARIABLE, None if settings is None else str(settings)),
        (AS_OF_VARIABLE, None if as_of is None else str(as_of.utc.isot)),
    ):
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)


def walk(node: Node) -> Iterator[Node]:
    """The elements under ``node``, depth first, in the page's order."""
    if isinstance(node, Block):
        for child in node.children.values():
            yield from walk(child)
    else:
        yield node


def shown(app: AppTest) -> str:
    """Every text of the page's main area, one per line; the data tables
    left out (a table's value is its data, not text)."""
    return "\n".join(
        str(getattr(node, "value", ""))
        for node in walk(app.main)
        if not isinstance(node, Dataframe)
    )


def downgrade_store(path: Path, revision: str) -> None:
    """Take the store at ``path`` down to ``revision`` of its schema, as an
    older keepout left it."""
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            command.downgrade(store_db._config(connection), revision)
    finally:
        engine.dispose()
