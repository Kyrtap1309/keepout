"""What the dashboard's pages share: the context, the time the data are seen
at, risk chips, provenance badges, errors and job status.

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
from typing import TYPE_CHECKING, Final, Literal

import streamlit as st

from keepout._text import utc
from keepout.dashboard.context import DashboardContext
from keepout.dashboard.text import literal
from keepout.errors import KeepoutError
from keepout.figures import ACTION_LABELS, RISK_COLOURS, RISK_LABELS, Provenance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from keepout.risk import Action, RiskLevel
    from keepout.store import Job

__all__ = [
    "CONTEXT_KEY",
    "JOB_REFRESH_SECONDS",
    "NOT_COMPUTED",
    "as_of_header",
    "ink",
    "job_status",
    "page_context",
    "provenance_badge",
    "risk_chip",
    "show_error",
]

CONTEXT_KEY: Final = "context"
"""The key of the :class:`~keepout.dashboard.context.DashboardContext` in
``st.session_state``, set by the app before a page runs."""

JOB_REFRESH_SECONDS: Final = 2
"""How often [s] the status of a queued or running job is read again."""

NOT_COMPUTED: Final = "not computed"
"""The chip of an event without a Pc for the settings."""

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

_PROVENANCE_HELP: Final[Mapping[Provenance, str]] = MappingProxyType(
    {
        Provenance.CDM_COVARIANCE: (
            "The Pc uses the covariances the CDM gives, from orbit determination."
        ),
        Provenance.GP_AGE_MODEL: (
            "The covariances are assumed by keepout's synthetic operator from "
            "the age of the GP data, not from orbit determination."
        ),
        Provenance.ASSUMED: (
            "The CDM marks its covariances as assumed (COVARIANCE_METHOD = "
            "DEFAULT), from a model it does not name."
        ),
        Provenance.MAX_PC: ("A maximum Pc: an upper bound found without a covariance."),
    }
)
"""What each provenance means, shown on hovering its badge."""

_CHIP_STYLE: Final = (
    "display:inline-block;padding:0.1rem 0.6rem;border-radius:999px;"
    "border:1px solid rgba(128,128,128,0.7);font-weight:600;font-size:0.9rem;"
)
"""The pill of a risk chip; its fill and text colour are added."""


def page_context() -> DashboardContext:
    """The context the app put in ``st.session_state`` for the page.

    Returns
    -------
    DashboardContext
        The store, settings and time of the dashboard. If there is none (a
        page run without the app), the page shows how to start the
        dashboard and stops (``st.stop``).
    """
    context = st.session_state.get(CONTEXT_KEY)
    if not isinstance(context, DashboardContext):
        st.error("No dashboard context: start it with keepout dashboard --store FILE")
        st.stop()
    return context


def as_of_header(ctx: DashboardContext) -> None:
    """The time the data are seen at, to the second in UTC, marked
    ``(fixed)`` when given rather than read from the clock."""
    stamp = utc(ctx.as_of)
    fixed = " (fixed)" if ctx.as_of_fixed else ""
    st.markdown(f"Data as of {stamp} UTC{fixed}")


def provenance_badge(p: Provenance) -> None:
    """A badge naming what a Pc rests on, with its meaning on hover."""
    st.badge(p.value, color=_PROVENANCE_COLOURS[p], help=_PROVENANCE_HELP[p])


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


def show_error(error: KeepoutError) -> None:
    """A keepout error as the page shows it: its message, verbatim
    (:func:`~keepout.dashboard.text.literal`), no traceback."""
    st.error(literal(str(error)))


def _read_jobs(store_path: Path, job_ids: Sequence[int]) -> list[Job]:
    """The jobs ``job_ids`` as they are now.

    Raises
    ------
    StoreError
        If the store cannot be opened or a job is not in it.
    """
    from keepout.store import JobQueue, open_store

    with open_store(store_path, create=False) as store:
        queue = JobQueue(store)
        return [queue.job(job_id) for job_id in job_ids]


_REFUSALS_KEY: Final = "job-cancel-refusals"
"""The session-state key of the refused cancels not yet seen through: job
id to the error."""

_POLLED_KEY: Final = "job-status-polled"
"""The session-state key of the jobs whose polling reran the page: the
next run is the polling's doing, not the user's."""


def _refusals() -> dict[int, KeepoutError]:
    """The refused cancels kept for the page, by job id."""
    kept = st.session_state.get(_REFUSALS_KEY)
    return dict(kept) if isinstance(kept, dict) else {}


def _polled(job_ids: Sequence[int]) -> None:
    """Mark the next run of the page as the polling's doing for ``job_ids``."""
    polled = st.session_state.get(_POLLED_KEY)
    st.session_state[_POLLED_KEY] = (
        set(polled) if isinstance(polled, set) else set()
    ) | set(job_ids)


def _cancel(store_path: Path, job_id: int) -> bool:
    """Cancel a job; if it cannot be (it has ended meanwhile, the store
    cannot be used), keep why for the page.

    Returns
    -------
    bool
        Whether the job was cancelled.
    """
    from keepout.store import JobQueue, open_store

    try:
        with open_store(store_path, create=False) as store:
            JobQueue(store).cancel(job_id)
    except KeepoutError as exc:
        st.session_state[_REFUSALS_KEY] = {**_refusals(), job_id: exc}
        return False
    return True


def _job_rows(
    store_path: Path,
    jobs: Sequence[Job],
    job_ids: Sequence[int],
    labels: Mapping[int, str],
) -> None:
    """A line per job: its kind and status with its label under them, a
    Cancel button while it is active, the error if it failed, and why a
    cancel of it was refused.

    A cancel reruns the page: as the user's doing if it is done, and as the
    polling's if it is refused, which keeps the refusal shown until the
    user's next interaction (the job has ended: the next poll reruns the
    page anyway).
    """
    refused = _refusals()
    for job in jobs:
        text, button = st.columns([5, 1], vertical_alignment="center")
        text.markdown(f"Job {job.id} ({job.kind.value}): **{job.status.value}**")
        if job.id in labels:
            text.caption(labels[job.id])
        if job.status.is_active and button.button("Cancel", key=f"cancel-job-{job.id}"):
            if not _cancel(store_path, job.id):
                _polled(job_ids)
            st.rerun()
        if job.error:
            st.error(literal(job.error))
        if job.id in refused:
            show_error(refused[job.id])


def _error_key(job_ids: Sequence[int]) -> str:
    """The session-state key of the store error that stopped the polling of
    ``job_ids``."""
    return "job-status-error-" + ",".join(map(str, job_ids))


@st.fragment(run_every=JOB_REFRESH_SECONDS)
def _live_jobs(
    store_path: Path, job_ids: Sequence[int], labels: Mapping[int, str]
) -> None:
    """The jobs' status, read again every :data:`JOB_REFRESH_SECONDS`. Once
    none is active, or the store cannot be read, the whole page runs again,
    without this fragment (the error is kept for :func:`job_status`)."""
    try:
        jobs = _read_jobs(store_path, job_ids)
    except KeepoutError as exc:
        st.session_state[_error_key(job_ids)] = exc
        _polled(job_ids)
        st.rerun()
    _job_rows(store_path, jobs, job_ids, labels)
    if not any(job.status.is_active for job in jobs):
        _polled(job_ids)
        st.rerun()


def _seen_refusals(job_ids: Sequence[int]) -> None:
    """Keep the refused cancels of jobs whose polling reran the page, drop
    the others (the user has acted since), and clear the marks; show those
    of jobs not among ``job_ids`` (the page no longer shows the job)."""
    polled = st.session_state.pop(_POLLED_KEY, None)
    marked = set(polled) if isinstance(polled, set) else set()
    kept = {job_id: exc for job_id, exc in _refusals().items() if job_id in marked}
    st.session_state[_REFUSALS_KEY] = kept
    for job_id, exc in kept.items():
        if job_id not in job_ids:
            show_error(exc)


def job_status(
    store_path: Path,
    job_ids: Sequence[int],
    *,
    labels: Mapping[int, str] | None = None,
) -> None:
    """The status of jobs of the store, with a Cancel button per queued or
    running one.

    While a job is queued or running, the status is read again every
    :data:`JOB_REFRESH_SECONDS` (an ``st.fragment``); once all have ended,
    it is shown once and no longer polled. A failed job's error is shown
    with ``st.error``. A store error is shown once with :func:`show_error`,
    and stops the polling until the page runs again (the next interaction
    tries again). A refused cancel's error stays, through the polls and
    the reruns they make, until the user's next interaction, even if the
    job is no longer among ``job_ids``. Call it once per page run (with no
    job, it shows only such errors).

    Parameters
    ----------
    store_path
        The store's file.
    job_ids
        The jobs to show, in order.
    labels
        A line under a job's status, by job id (e.g. its settings).
    """
    _seen_refusals(job_ids)
    if not job_ids:
        return
    named = dict(labels or {})
    stopped = st.session_state.pop(_error_key(job_ids), None)
    if isinstance(stopped, KeepoutError):
        show_error(stopped)
        return
    try:
        jobs = _read_jobs(store_path, job_ids)
    except KeepoutError as exc:
        show_error(exc)
        return
    if any(job.status.is_active for job in jobs):
        _live_jobs(store_path, tuple(job_ids), named)
    else:
        _job_rows(store_path, jobs, job_ids, named)
