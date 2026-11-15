"""The dashboard's app and its context: the store, settings and time it is
run on, from the environment ``keepout dashboard`` sets."""

import shutil
from pathlib import Path

import dashboard_seed as seed
import pytest
from astropy.time import Time
from dashboard_builders import OpenApp, downgrade_store
from streamlit.testing.v1 import AppTest

from keepout.dashboard.context import DashboardContext, context_from_env
from keepout.errors import SettingsError, StoreError
from keepout.settings import Settings, load_settings, settings_to_data

NOW = Time("2026-11-14T09:30:00", scale="utc")


def _markdown(app: AppTest) -> str:
    """Every Markdown element of the app's page, one per line."""
    return "\n".join(item.value for item in app.markdown)


def test_app_runs_and_shows_as_of(app_test: OpenApp) -> None:
    app = app_test()
    assert not app.exception
    assert not app.error
    shown = _markdown(app)
    assert "Data as of 2026-10-07T12:00:00 UTC (fixed)" in shown


@pytest.mark.parametrize(
    ("page", "title"), [("event", "Event"), ("planner", "Manoeuvre planner")]
)
def test_every_page_shows_as_of(app_test: OpenApp, page: str, title: str) -> None:
    app = app_test(page)
    assert not app.exception
    assert [item.value for item in app.title] == [title]
    assert "Data as of 2026-10-07T12:00:00 UTC (fixed)" in _markdown(app)


def test_app_on_old_schema(
    app_test: OpenApp, seed_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = Path(shutil.copy2(seed_db, tmp_path / "old.db"))
    downgrade_store(old, "0002")
    before = (old.read_bytes(), old.stat().st_mtime_ns)
    monkeypatch.setenv("KEEPOUT_STORE", str(old))
    app = app_test()
    assert not app.exception
    assert len(app.error) == 1
    assert "keepout store upgrade" in str(app.error[0].value)
    assert (old.read_bytes(), old.stat().st_mtime_ns) == before


def test_app_without_store_env(
    app_test: OpenApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KEEPOUT_STORE")
    app = app_test()
    assert not app.exception
    assert len(app.error) == 1
    assert "KEEPOUT_STORE" in str(app.error[0].value)


def test_app_with_a_bad_as_of(
    app_test: OpenApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEPOUT_AS_OF", "soon")
    app = app_test()
    assert not app.exception
    assert [str(item.value) for item in app.error] == [
        "KEEPOUT_AS_OF: not an ISO 8601 UTC time: 'soon'"
    ]


def test_app_shares_the_context_with_its_pages(app_test: OpenApp) -> None:
    app = app_test()
    context = app.session_state["context"]
    assert isinstance(context, DashboardContext)
    assert context.as_of_fixed
    assert context.as_of.utc.isot == "2026-10-07T12:00:00.000"


def test_context_from_env_defaults(tmp_path: Path) -> None:
    context = context_from_env(
        {"KEEPOUT_STORE": str(tmp_path / "keepout.db")}, now=lambda: NOW
    )
    assert context.store_path == tmp_path / "keepout.db"
    assert settings_to_data(context.settings) == settings_to_data(Settings())
    assert context.as_of is NOW
    assert not context.as_of_fixed


def test_context_from_env_given_everything(tmp_path: Path) -> None:
    context = context_from_env(
        {
            "KEEPOUT_STORE": str(tmp_path / "keepout.db"),
            "KEEPOUT_SETTINGS": str(seed.SETTINGS),
            "KEEPOUT_AS_OF": "2026-10-07T12:00",
        },
        now=lambda: NOW,
    )
    assert settings_to_data(context.settings) == settings_to_data(
        load_settings(seed.SETTINGS)
    )
    assert context.as_of.utc.isot == "2026-10-07T12:00:00.000"
    assert context.as_of_fixed


def test_context_from_env_reads_as_of_as_the_command_line_does(tmp_path: Path) -> None:
    # The command line's --as-of parser (keepout._environment.parse_time).
    context = context_from_env(
        {"KEEPOUT_STORE": str(tmp_path / "keepout.db"), "KEEPOUT_AS_OF": "2026-10-07"}
    )
    assert context.as_of.utc.isot == "2026-10-07T00:00:00.000"


def test_context_from_env_refuses_a_bad_as_of(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="KEEPOUT_AS_OF: not an ISO 8601 UTC time"):
        context_from_env(
            {"KEEPOUT_STORE": str(tmp_path / "keepout.db"), "KEEPOUT_AS_OF": "soon"}
        )


def test_context_from_env_needs_a_store() -> None:
    with pytest.raises(StoreError, match="KEEPOUT_STORE is not set"):
        context_from_env({})


def test_context_from_env_refuses_bad_settings(tmp_path: Path) -> None:
    bad = tmp_path / "settings.toml"
    bad.write_text("[pc]\nmethod = 'guess'\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="method"):
        context_from_env(
            {
                "KEEPOUT_STORE": str(tmp_path / "keepout.db"),
                "KEEPOUT_SETTINGS": str(bad),
            }
        )
