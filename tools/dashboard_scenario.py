"""The dashboard's demo scenario in a browser (Playwright): the events
list, the planned event's manoeuvre planner and page, the demo's update
CDM uploaded, and the event's Pc rising to red once the worker has
computed it.

It expects ``docker compose up`` on a fresh volume: the demo's store,
seen at :data:`dashboard_seed.DEMO_AS_OF`. The e2e test
(``tests/e2e``) runs it; ``tools/dashboard_gif.py`` records it, with
pauses for the viewer.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Final

from dashboard_seed import DEMO_AS_OF, DEMO_EVENT, ISSUES
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import ViewportSize, expect

from keepout._text import utc
from keepout.figures.palette import RISK_LABELS
from keepout.risk import RiskLevel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from playwright.sync_api import Locator, Page

AS_OF_TEXT: Final = f"Data as of {utc(DEMO_AS_OF)} UTC (fixed)"
"""How every page of the demo gives the time its data are seen at."""

RED: Final = re.compile(f"^{re.escape(RISK_LABELS[RiskLevel.RED])} · ")
"""A red risk chip: its level, then its action."""

VIEWPORT: Final = ViewportSize(width=1280, height=800)
"""The browser's size [px] for the scenario (and the GIF's video)."""

UPLOADED_SIZE: Final = re.compile(r"^\d+(\.\d+)?\s?[KM]?B$")
"""The size Streamlit lists under a file once it has it."""

POLL_S: Final = 2.0
"""How often [s] the scenario reloads a page while it waits for the worker."""


def _step(name: str, check: Callable[[], None]) -> None:
    """Run ``check``, naming the step in the error if it fails.

    Raises
    ------
    AssertionError
        If ``check`` fails, its error chained.
    """
    try:
        check()
    except (AssertionError, PlaywrightError) as exc:
        raise AssertionError(f"scenario step {name!r} failed: {exc}") from exc


def _visible(locator: Locator, timeout_s: float) -> None:
    """Wait until ``locator`` is visible.

    Raises
    ------
    AssertionError
        If it is not within ``timeout_s``.
    """
    expect(locator.first).to_be_visible(timeout=timeout_s * 1000)


def _reload_until(page: Page, locator: Locator, timeout_s: float) -> None:
    """Reload ``page`` every :data:`POLL_S` until ``locator`` is visible.

    Raises
    ------
    AssertionError
        If it is not within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _visible(locator, POLL_S)
        except AssertionError:
            if time.monotonic() > deadline:
                raise
            page.reload()
        else:
            return


def run_scenario(
    page: Page,
    base_url: str,
    *,
    upload: Path,
    pause_s: float = 0.0,
    timeout_s: float = 60.0,
) -> None:
    """Go through the demo in ``page``: the list, the planner, the event,
    the upload of ``upload``, and the event again, red.

    Parameters
    ----------
    page
        A browser page.
    base_url
        The dashboard's URL, without a trailing slash.
    upload
        The demo's update CDM (``examples/dashboard/demo/*.kvn``).
    pause_s
        How long [s] to hold each view, for a recording.
    timeout_s
        How long [s] to wait for each view, and for the worker.

    Raises
    ------
    AssertionError
        If a step fails, naming it.
    """
    event_url = f"{base_url}/event?event={DEMO_EVENT}"

    def hold() -> None:
        if pause_s:
            page.wait_for_timeout(pause_s * 1000)

    def show(locator: Locator) -> None:
        """Hold the view, then scroll to ``locator`` and hold it too."""
        hold()
        locator.first.scroll_into_view_if_needed()
        hold()

    def events_list() -> None:
        page.goto(base_url)
        _visible(page.get_by_text(AS_OF_TEXT), timeout_s)
        _visible(page.get_by_role("link", name=DEMO_EVENT), timeout_s)
        _visible(page.get_by_text("assumed (GP age model)"), timeout_s)

    def planner() -> None:
        page.goto(f"{base_url}/planner?event={DEMO_EVENT}")
        _visible(page.get_by_text("Pc over burn time and Δv"), timeout_s)
        _visible(page.locator(".js-plotly-plot"), timeout_s)
        _visible(page.get_by_text("Smallest burn each way"), timeout_s)
        _visible(page.get_by_text("Re-screen"), timeout_s)
        show(page.locator(".js-plotly-plot"))  # The Pc heatmap over burn time and Δv.

    def event_before() -> None:
        page.goto(event_url)
        _visible(page.get_by_text(f"{ISSUES} CDMs"), timeout_s)
        _visible(page.get_by_text("Pc trend"), timeout_s)
        show(page.locator(".js-plotly-plot"))  # The trend, 4 yellow points.

    def uploaded() -> None:
        page.goto(base_url)
        page.locator("input[type=file]").set_input_files(upload)
        # Streamlit sends the file in the background: Ingest before it is
        # listed with its size would submit no file. (The name is shortened.)
        uploader = page.get_by_test_id("stFileUploader")
        _visible(uploader.get_by_text(UPLOADED_SIZE), timeout_s)
        hold()
        page.get_by_role("button", name="Ingest").click()
        _visible(page.get_by_text("Stored 1 CDM (0 already stored)"), timeout_s)

    def event_after() -> None:
        page.goto(event_url)
        _visible(page.get_by_text(f"{ISSUES + 1} CDMs"), timeout_s)
        _reload_until(page, page.get_by_text(RED), timeout_s)
        show(page.locator(".js-plotly-plot"))  # The trend, a 5th point, red.
        hold()

    for name, check in (
        ("events list", events_list),
        ("planner", planner),
        ("event before the update", event_before),
        ("upload", uploaded),
        ("event after the update", event_after),
    ):
        _step(name, check)
        hold()
