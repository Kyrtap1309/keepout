"""The dashboard's entry script, which ``keepout dashboard`` runs with
``streamlit run``.

It reads the context from the environment
(:func:`~keepout.dashboard.context.context_from_env`; the settings file is
read again only once it has changed), opens the store the pages share
(:func:`~keepout.dashboard.widgets.shared_store`, which checks it is at
this keepout's schema), puts the context in ``st.session_state`` for the
pages and runs the page chosen in the navigation, under the time the data
are seen at. An error is shown as its message, and the page stops.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from keepout.dashboard.context import DashboardContext, context_from_env
from keepout.dashboard.routes import CONTEXT_KEY, EVENT_PAGE, EVENTS_PAGE, PLANNER_PAGE
from keepout.dashboard.widgets import as_of_header, shared_store, show_error
from keepout.errors import KeepoutError
from keepout.settings import Settings, load_settings


@st.cache_data(show_spinner=False, max_entries=8)
def _settings_of(path: str, mtime_ns: int) -> Settings:  # noqa: ARG001 (the key)
    """The settings file at ``path`` as it was at ``mtime_ns``."""
    return load_settings(Path(path))


def _read_settings(path: Path) -> Settings:
    """The settings file, read again only once it has changed."""
    return _settings_of(str(path), path.stat().st_mtime_ns)


def _context() -> DashboardContext:
    """The context of this run, its store opened; on an error, the message
    and a stopped page."""
    try:
        context = context_from_env(read_settings=_read_settings)
        # Refused without a change to the file if made by an older keepout.
        shared_store(context.store_path)
    except (KeepoutError, ValueError, OSError) as exc:
        show_error(exc)
        st.stop()
    return context


def main() -> None:
    """Run the page chosen in the navigation, after the context and the
    time the data are seen at."""
    st.set_page_config(page_title="keepout", layout="wide")
    context = _context()
    st.session_state[CONTEXT_KEY] = context
    page = st.navigation(
        [
            st.Page(EVENTS_PAGE, title="Events", default=True),
            # Each shows the event in its address: reached from the list's
            # links, not the navigation.
            st.Page(EVENT_PAGE, title="Event", visibility="hidden"),
            st.Page(PLANNER_PAGE, title="Manoeuvre planner", visibility="hidden"),
        ]
    )
    as_of_header(context)
    page.run()


main()
