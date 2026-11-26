"""The dashboard in a browser, as ``docker compose up`` serves it: the
scenario the README's GIF records (``tools/dashboard_scenario.py``)."""

import dashboard_seed as seed
import pytest
from dashboard_scenario import run_scenario
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e


def test_dashboard_scenario(dashboard_url: str, page: Page) -> None:
    # dashboard_url first: it skips before a browser is launched.
    run_scenario(page, dashboard_url, upload=seed.demo_file())
