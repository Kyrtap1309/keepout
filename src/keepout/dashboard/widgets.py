"""What the dashboard's pages share: the context and the session's state,
the time the data are seen at, tables, risk chips, provenance badges,
errors and job status.

Risk is shown as colour and words together: a chip filled with the level's
colour (:data:`~keepout.figures.RISK_COLOURS`) carries its label and the
action's, in black or white, whichever contrasts more with the fill, and
has a grey outline. The fill, not the page, is behind the text, so the chip
reads the same on Streamlit's light and dark themes (the dark red of
:attr:`~keepout.risk.RiskLevel.RED` gets white text).
"""

from __future__ import annotations

import html
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeVar

import streamlit as st

from keepout._text import utc
from keepout.dashboard.context import DashboardContext
from keepout.dashboard.routes import CONTEXT_KEY, cancel_job_key
from keepout.dashboard.text import NOT_COMPUTED, literal, unrated
from keepout.errors import KeepoutError
from keepout.figures import (
    ACTION_LABELS,
    PROVENANCE_HELP,
    PROVENANCE_LABELS,
    RISK_COLOURS,
    RISK_LABELS,
)
from keepout.risk import Provenance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from streamlit.delta_generator import DeltaGenerator

    from keepout.risk import Action, Assessment, RiskLevel
    from keepout.store import Job

__all__ = [
    "JOB_REFRESH_SECONDS",
    "as_of_header",
    "assessment_chip",
    "ink",
    "job_status",
    "page_context",
    "provenance_badge",
    "risk_chip",
    "show_error",
    "state",
    "table_header",
    "table_row",
]

JOB_REFRESH_SECONDS: Final = 2
"""How often [s] the status of a queued or running job is read again."""

_T = TypeVar("_T")

_BadgeColour = Literal["blue", "orange", "violet", "gray"]

_PROVENANCE_COLOURS: Final[Mapping[Provenance, _BadgeColour]] = MappingProxyType(
    {
        Provenance.CDM_COVARIANCE: "blue",
        Provenance.GP_AGE_MODEL: "orange",
        Provenance.ASSUMED: "orange",
        Provenance.MAX_PC: "violet",
    }
)
"""The colour of each provenance badge; the badge names it in words."""

_CHIP_STYLE: Final = (
    "display:inline-block;padding:0.1rem 0.6rem;border-radius:999px;"
    "border:1px solid rgba(128,128,128,0.7);font-weight:600;font-size:0.9rem;"
)
"""The pill of a risk chip; its fill and text colour are added."""


def state(key: str, kind: type[_T]) -> _T | None:
    """What ``st.session_state`` keeps under ``key``, if it is a ``kind``;
    else ``None``."""
    value = st.session_state.get(key)
    return value if isinstance(value, kind) else None


def page_context() -> DashboardContext:
    """The context the app put in ``st.session_state`` for the page.

    Returns
    -------
    DashboardContext
        The store, settings and time of the dashboard. If there is none (a
        page run without the app), the page shows how to start the
        dashboard and stops (``st.stop``).
    """
    context = state(CONTEXT_KEY, DashboardContext)
    if context is None:
        st.error("No dashboard context: start it with keepout dashboard --store FILE")
        st.stop()
    return context


def as_of_header(ctx: DashboardContext) -> None:
    """The time the data are seen at, to the second in UTC, marked
    ``(fixed)`` when given rather than read from the clock."""
    stamp = utc(ctx.as_of)
    fixed = " (fixed)" if ctx.as_of_fixed else ""
    st.markdown(f"Data as of {stamp} UTC{fixed}")


def table_header(columns: Sequence[tuple[str, int]]) -> None:
    """The titles of a table's columns, each ``(title, width)``, in bold."""
    cells = st.columns([width for _, width in columns])
    for cell, (title, _) in zip(cells, columns, strict=True):
        cell.markdown(f"**{title}**")


def table_row(columns: Sequence[tuple[str, int]]) -> list[DeltaGenerator]:
    """The cells of a row of a table of ``columns`` (as
    :func:`table_header`), their content centred vertically."""
    return st.columns([width for _, width in columns], vertical_alignment="center")


def provenance_badge(p: Provenance) -> None:
    """A badge naming what a Pc rests on, with its meaning on hover."""
    st.badge(
        PROVENANCE_LABELS[p], color=_PROVENANCE_COLOURS[p], help=PROVENANCE_HELP[p]
    )


def _luminance(colour: str) -> float:
    """The relative luminance (WCAG 2) of a ``#rrggbb`` colour."""
    channels = [int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def ink(fill: str) -> str:
    """Black or white, whichever contrasts more with ``fill`` (``#rrggbb``)."""
    lum = _luminance(fill)
    on_black = (lum + 0.05) / 0.05
    on_white = 1.05 / (lum + 0.05)
    return "#000000" if on_black >= on_white else "#ffffff"


def risk_chip(
    level: RiskLevel | None, action: Action | None, *, missing: str = NOT_COMPUTED
) -> None:
    """An event's risk as a chip: its level's colour with the level and the
    action in words; ``None`` is an outlined chip in the theme's text colour
    saying why there is no level, ``missing`` (default: no Pc for the
    settings, "not computed")."""
    if level is None:
        style = f"{_CHIP_STYLE}border-style:dashed;color:inherit;"
        text = missing
    else:
        fill = RISK_COLOURS[level]
        style = f"{_CHIP_STYLE}background:{fill};color:{ink(fill)};"
        text = RISK_LABELS[level]
        if action is not None:
            text += f" · {ACTION_LABELS[action]}"
    st.markdown(
        f'<span style="{style}">{html.escape(text)}</span>', unsafe_allow_html=True
    )


def assessment_chip(assessment: Assessment | None) -> None:
    """An event's risk as a chip (:func:`risk_chip`): its level and action;
    without a level, why (:func:`~keepout.dashboard.text.unrated`); without
    an assessment, not computed."""
    if assessment is None:
        risk_chip(None, None)
    elif assessment.level is None:
        risk_chip(None, None, missing=unrated(assessment))
    else:
        risk_chip(assessment.level, assessment.decision.action)


def show_error(error: KeepoutError | ValueError) -> None:
    """An error as the page shows it: its message, verbatim
    (:func:`~keepout.dashboard.text.literal`), no traceback."""
    st.error(literal(str(error)))


def _read_jobs(store_path: Path, job_ids: Sequence[int]) -> list[Job]:
    """The jobs ``job_ids`` as they are now, read in one transaction.

    Raises
    ------
    StoreError
        If the store cannot be opened or a job is not in it.
    """
    from keepout.store import JobQueue, open_store

    with open_store(store_path, create=False) as store:
        return JobQueue(store).jobs_by_id(job_ids)


def _cancel(store_path: Path, job_id: int) -> None:
    """Cancel a job; if it cannot be (it has ended meanwhile, the store
    cannot be used), say why in a toast, which outlives the rerun that
    follows and is not kept for another run or page."""
    from keepout.store import JobQueue, open_store

    try:
        with open_store(store_path, create=False) as store:
            JobQueue(store).cancel(job_id)
    except KeepoutError as exc:
        st.toast(literal(str(exc)), icon=":material/error:")


def _job_rows(store_path: Path, jobs: Sequence[Job], labels: Mapping[int, str]) -> None:
    """A line per job: its kind and status with its label under them, a
    Cancel button while it is active, and the error if it failed. A cancel,
    done or refused, reruns the page, which shows the jobs as they are
    then."""
    for job in jobs:
        text, button = st.columns([5, 1], vertical_alignment="center")
        text.markdown(f"Job {job.id} ({job.kind.value}): **{job.status.value}**")
        if job.id in labels:
            text.caption(labels[job.id])
        if job.status.is_active and button.button("Cancel", key=cancel_job_key(job.id)):
            _cancel(store_path, job.id)
            st.rerun()
        if job.error:
            st.error(literal(job.error))


@st.fragment(run_every=JOB_REFRESH_SECONDS)
def _live_jobs(
    store_path: Path,
    job_ids: Sequence[int],
    labels: Mapping[int, str],
    first: list[list[Job]],
) -> None:
    """The jobs' status, read again every :data:`JOB_REFRESH_SECONDS`. Once
    none is active, the whole page runs again, without this fragment. If
    the store cannot be read, the error shows in place of the jobs until a
    later read succeeds.

    ``first`` holds the jobs the page has just read, which the fragment's
    first run shows and takes out; its later runs (the fragment keeps its
    arguments) read them again.
    """
    try:
        jobs = first.pop() if first else _read_jobs(store_path, job_ids)
    except KeepoutError as exc:
        show_error(exc)
        return
    _job_rows(store_path, jobs, labels)
    if not any(job.status.is_active for job in jobs):
        st.rerun()


def job_status(
    store_path: Path,
    job_ids: Sequence[int],
    *,
    labels: Mapping[int, str] | None = None,
) -> bool:
    """The status of jobs of the store, with a Cancel button per queued or
    running one.

    While a job is queued or running, the status is read again every
    :data:`JOB_REFRESH_SECONDS` (an ``st.fragment``); once all have ended,
    it is shown once and no longer polled. A failed job's error is shown
    with ``st.error``, a store error with :func:`show_error` (while
    polling, until a later read succeeds) and a refused cancel's error in
    a toast.

    Parameters
    ----------
    store_path
        The store's file.
    job_ids
        The jobs to show, in order.
    labels
        A line under a job's status, by job id (e.g. its settings).

    Returns
    -------
    bool
        Whether a job is queued or running, as read now; ``False`` if the
        store cannot be read.
    """
    if not job_ids:
        return False
    named = dict(labels or {})
    try:
        jobs = _read_jobs(store_path, job_ids)
    except KeepoutError as exc:
        show_error(exc)
        return False
    if any(job.status.is_active for job in jobs):
        _live_jobs(store_path, tuple(job_ids), named, [jobs])
        return True
    _job_rows(store_path, jobs, named)
    return False
