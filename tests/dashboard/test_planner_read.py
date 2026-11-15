"""What the manoeuvre planner page reads of the store: an event, its cached
plan and its jobs, as they were at one moment."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import dashboard_seed as seed
import pytest
from dashboard_builders import APP, TIMEOUT, use_store
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from keepout.settings import load_settings
from keepout.store import Store, open_store


def test_the_plan_is_read_in_one_transaction(
    seed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A CDM ingested between reads would show a plan or a job of another
    # CDM than the event's latest: the event, its plan and its jobs are
    # read in one transaction.
    use_store(monkeypatch, seed_copy, seed.SETTINGS, seed.seed_as_of())
    with open_store(seed_copy, create=False) as store:
        page = store.summaries(load_settings(seed.SETTINGS), as_of=seed.seed_as_of())
    event_id = page.items[0].event_id
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    app.switch_page("views/planner.py")
    app.query_params.update({"event": event_id})
    opened: list[None] = []
    read = Store.read

    @contextmanager
    def counted(self: Store) -> Generator[Session]:
        opened.append(None)
        with read(self) as session:
            yield session

    monkeypatch.setattr(Store, "read", counted)
    app.run()
    assert not app.exception
    assert [item.value for item in app.title] == [
        f"Manoeuvre planner: event {event_id}"
    ]
    assert len(opened) == 1
