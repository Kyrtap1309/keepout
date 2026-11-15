"""The dashboard's entry script, which ``keepout dashboard`` runs with
``streamlit run``.

It reads the context from the environment
(:func:`~keepout.dashboard.context.context_from_env`), checks that the store
opens at this keepout's schema, puts the context in ``st.session_state``
for the pages and runs the page chosen in the navigation, under the time the
data are seen at. An error is shown as its message, and the page stops.
"""

from __future__ import annotations

import streamlit as st

from keepout.dashboard.context import DashboardContext, context_from_env
from keepout.dashboard.routes import CONTEXT_KEY, EVENT_PAGE, EVENTS_PAGE, PLANNER_PAGE
from keepout.dashboard.text import literal
from keepout.dashboard.widgets import as_of_header, show_error
from keepout.errors import KeepoutError


def _context() -> DashboardContext:
    """The context of this run, its store checked to open; on an error, the
    message and a stopped page."""
    try:
        context = context_from_env()
        # Refused without a change to the file if made by an older keepout.
        context.store().close()
    except (KeepoutError, ValueError) as exc:
        show_error(exc)
        st.stop()
    except OSError as exc:
        st.error(literal(str(exc)))
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
            st.Page(EVENT_PAGE, title="Event"),
            st.Page(PLANNER_PAGE, title="Manoeuvre planner"),
        ]
    )
    as_of_header(context)
    page.run()


main()
