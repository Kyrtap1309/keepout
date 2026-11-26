"""The e2e tests drive the dashboard at ``--base-url`` (``docker compose up``
on a fresh volume) in pytest-playwright's browser; without it they are
skipped."""

import pytest
from dashboard_scenario import VIEWPORT


@pytest.fixture
def dashboard_url(base_url: str | None) -> str:
    if not base_url:
        pytest.skip("give --base-url a running dashboard (docker compose up)")
    return base_url.rstrip("/")


@pytest.fixture(scope="session")
def browser_context_args(
    browser_context_args: dict[str, object],
) -> dict[str, object]:
    return {**browser_context_args, "viewport": VIEWPORT}
