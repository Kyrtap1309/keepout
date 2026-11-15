"""The dashboard's entry script, which ``keepout dashboard`` runs with
``streamlit run``.

It reads the context from the environment
(:func:`~keepout.dashboard.context.context_from_env`), checks that the store
opens at this keepout's schema, puts the context in ``st.session_state``
for the pages and runs the page chosen in the navigation, under the time the
data are seen at. An error is shown as its message, and the page stops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import streamlit as st

from keepout.dashboard.context import DashboardContext, context_from_env
from keepout.dashboard.text import literal
from keepout.dashboard.widgets import CONTEXT_KEY, as_of_header, show_error
from keepout.errors import KeepoutError

# Not "pages": Streamlit would take a pages folder beside the entry script
# for its older multipage apps and run a page's script without this one.
# (No docstring: Streamlit would show a bare string here as Markdown.)
VIEWS: Final = Path(__file__).parent / "views"


def _context() -> DashboardContext:
    """The context of this run, its store checked to open; on an error, the
    message and a stopped page."""
    from keepout.store import open_store

    try:
        context = context_from_env()
        # Refused without a change to the file if made by an older keepout.
        open_store(context.store_path, create=False).close()
    except KeepoutError as exc:
        show_error(exc)
        st.stop()
    except (ValueError, OSError) as exc:
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
            st.Page(VIEWS / "events.py", title="Events", default=True),
            st.Page(VIEWS / "event.py", title="Event"),
            st.Page(VIEWS / "planner.py", title="Manoeuvre planner"),
        ]
    )
    as_of_header(context)
    page.run()


main()
