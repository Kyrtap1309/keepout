"""Where the dashboard's pages are and what they keep: the page scripts, the
query parameter naming an event, and the keys of ``st.session_state``.

Not a page: importing it runs nothing (it does not import Streamlit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "CONTEXT_KEY",
    "EVENTS_PAGE",
    "EVENT_PAGE",
    "EVENT_PARAM",
    "OFFSET_KEY",
    "PAST_KEY",
    "PLANNER_PAGE",
    "UPLOAD_KEY",
    "VIEWS",
    "cancel_job_key",
    "refresh_job_key",
    "what_if_key",
]

VIEWS: Final = Path(__file__).with_name("views")
"""The folder of the page scripts. Not ``pages``: Streamlit would take a
pages folder beside the entry script for its older multipage apps and run
a page's script without the app."""

EVENTS_PAGE: Final = VIEWS / "events.py"
"""The events list, the app's default page."""

EVENT_PAGE: Final = VIEWS / "event.py"
"""An event's page."""

PLANNER_PAGE: Final = VIEWS / "planner.py"
"""An event's manoeuvre planner."""

EVENT_PARAM: Final = "event"
"""The query parameter of the event and planner pages: the event's id."""

CONTEXT_KEY: Final = "context"
"""The key of the :class:`~keepout.dashboard.context.DashboardContext`, set
by the app before a page runs."""

OFFSET_KEY: Final = "events-offset"
"""The key of the events list's offset: the events before its page."""

PAST_KEY: Final = "events-past"
"""The key of the events list's toggle of the past events."""

UPLOAD_KEY: Final = "events-upload"
"""The key of the last upload's outcome, kept for the events page."""


def refresh_job_key(event_id: str) -> str:
    """The key of the ``refresh`` job the session queued for an event."""
    return f"event-refresh-{event_id}"


def what_if_key(event_id: str) -> str:
    """The key of an event's what-if on the planner: the settings chosen
    for it."""
    return f"planner-chosen-{event_id}"


def cancel_job_key(job_id: int) -> str:
    """The key of a job's Cancel button."""
    return f"cancel-job-{job_id}"
