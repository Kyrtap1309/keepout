"""The dashboard in a browser, as ``docker compose up`` serves it: the
scenario the README's GIF records (``tools/dashboard_scenario.py``)."""

import dashboard_seed as seed
import pytest
from dashboard_scenario import run_scenario
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e


def test_dashboard_scenario(page: Page, base_url: str) -> None:
    [upload] = sorted(seed.DEMO_DIR.glob("*.kvn"))
    run_scenario(page, base_url, upload=upload)
