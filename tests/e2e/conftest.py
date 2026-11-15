"""The e2e tests' browser and the dashboard they drive: the one at
``KEEPOUT_E2E_URL`` (``docker compose up`` on a fresh volume); without it
they are skipped."""

import os
from collections.abc import Iterator

import pytest
from playwright.sync_api import Page, sync_playwright

URL_VARIABLE = "KEEPOUT_E2E_URL"


@pytest.fixture
def base_url() -> str:
    url = os.environ.get(URL_VARIABLE)
    if not url:
        pytest.skip(f"set {URL_VARIABLE} to a running dashboard (docker compose up)")
    return url.rstrip("/")


@pytest.fixture
def page() -> Iterator[Page]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            yield browser.new_page(viewport={"width": 1280, "height": 800})
        finally:
            browser.close()
