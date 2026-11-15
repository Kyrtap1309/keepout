"""The events page: the stored events, a page at a time, with their risk, and
the upload of CDM files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import streamlit as st

from keepout._text import pc_text, plural, utc
from keepout.dashboard import text
from keepout.dashboard.listing import PAGE_SIZE, read_page
from keepout.dashboard.upload import (
    MAX_FILE_BYTES,
    MAX_FILES,
    UploadResult,
    ingest_upload,
)
from keepout.dashboard.widgets import (
    NOT_COMPUTED,
    job_status,
    page_context,
    provenance_badge,
    risk_chip,
    show_error,
)
from keepout.errors import KeepoutError

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.dashboard.context import DashboardContext
    from keepout.store import EventSummary, IngestReport, SummaryPage

# (No docstrings on the names below: Streamlit would show a bare string as
# Markdown.)
EVENT_PAGE: Final = Path(__file__).with_name("event.py")
PLANNER_PAGE: Final = Path(__file__).with_name("planner.py")
EVENT_PARAM: Final = "event"
OFFSET_KEY: Final = "events-offset"
PAST_KEY: Final = "events-past"
UPLOAD_KEY: Final = "events-upload"
COLUMNS: Final = (
    ("Event", 3),
    ("Objects", 3),
    ("TCA (UTC)", 3),
    ("Until TCA", 2),
    ("Latest Pc", 2),
    ("Provenance", 3),
    ("Risk", 3),
    ("CDMs", 1),
    ("Job", 1),
    ("Manoeuvre", 2),
)


def _upload_message(report: IngestReport, jobs: int) -> str:
    """What an upload stored and queued, in words."""
    return (
        f"Stored {plural(len(report.added), 'CDM')} "
        f"({len(report.already_stored)} already stored) in "
        f"{plural(len(report.events), 'event')}; queued "
        f"{plural(jobs, 'Pc refresh job')}."
    )


def _ingest(ctx: DashboardContext, files: list[tuple[str, bytes]]) -> None:
    """Ingest uploaded files, the outcome kept in the session for
    :func:`_last_upload`; a store that cannot be opened shows its error."""
    from keepout.store import open_store

    st.session_state.pop(UPLOAD_KEY, None)
    if not files:
        st.warning("Choose CDM files to ingest.")
        return
    try:
        with open_store(ctx.store_path, create=False) as store:
            st.session_state[UPLOAD_KEY] = ingest_upload(store, ctx.settings, files)
    except KeepoutError as exc:
        show_error(exc)


def _any_active(ctx: DashboardContext, job_ids: tuple[int, ...]) -> bool:
    """Whether any of the jobs is queued or running; ``False`` if the store
    cannot be read (the job status shows why)."""
    from keepout.store import JobQueue, open_store

    try:
        with open_store(ctx.store_path, create=False) as store:
            queue = JobQueue(store)
            return any(queue.job(job_id).status.is_active for job_id in job_ids)
    except KeepoutError:
        return False


def _last_upload(ctx: DashboardContext) -> None:
    """The outcome of the session's last upload: its errors, each naming its
    file, or what it stored and the status of the jobs it queued. It is
    shown once, and again only while a job it queued is queued or running."""
    result = st.session_state.pop(UPLOAD_KEY, None)
    if not isinstance(result, UploadResult):
        return
    for name, message in result.errors:
        shown = text.literal(message)
        st.error(f"{text.literal(name)}: {shown}" if name else shown)
    if result.report is not None:
        st.success(_upload_message(result.report, len(result.jobs)))
    if result.jobs:
        job_status(ctx.store_path, result.jobs)
        if _any_active(ctx, result.jobs):
            st.session_state[UPLOAD_KEY] = result


def _upload(ctx: DashboardContext) -> None:
    """The upload form: CDM files ingested, their events' missing Pc queued;
    the outcome of the last upload (:func:`_last_upload`)."""
    with st.expander("Upload CDMs"):
        with st.form("upload", clear_on_submit=True):
            uploaded = st.file_uploader(
                f"CDM files (KVN or XML, gzip-compressed or not; at most "
                f"{MAX_FILES} files of {MAX_FILE_BYTES // 1_000_000} MB each)",
                accept_multiple_files=True,
            )
            submitted = st.form_submit_button("Ingest")
        if submitted:
            _ingest(ctx, [(file.name, file.getvalue()) for file in uploaded or []])
        _last_upload(ctx)


def _read_page(ctx: DashboardContext, *, include_past: bool) -> SummaryPage:
    """The page of the list at the offset kept in the session, moved back to
    the last page if the list has become shorter.

    Raises
    ------
    StoreError
        If the store cannot be opened or read.
    """
    page = read_page(
        ctx.store_path,
        ctx.settings,
        as_of=ctx.as_of,
        include_past=include_past,
        offset=int(st.session_state.get(OFFSET_KEY, 0)),
    )
    st.session_state[OFFSET_KEY] = page.offset
    return page


def _header() -> None:
    """The column titles of the list."""
    for column, (title, _) in zip(
        st.columns([width for _, width in COLUMNS]), COLUMNS, strict=True
    ):
        column.markdown(f"**{title}**")


def _row(summary: EventSummary, as_of: Time) -> None:
    """An event of the list, its id a link to its page; a link to its
    manoeuvre planner."""
    cells = st.columns([width for _, width in COLUMNS], vertical_alignment="center")
    event, objects, tca, countdown, pc, origin, risk, cdms, job, plan = cells
    with event:
        st.page_link(
            EVENT_PAGE,
            label=summary.event_id,
            query_params={EVENT_PARAM: summary.event_id},
        )
    objects.markdown(text.literal(f"{summary.object1} / {summary.object2}"))
    tca.markdown(utc(summary.tca))
    countdown.markdown(text.countdown(summary.tca, as_of))
    latest = summary.latest
    if latest is None:
        pc.markdown(NOT_COMPUTED)
    elif latest.pc is None:
        pc.markdown(text.FAILED)
    else:
        pc.markdown(pc_text(latest.pc))
    if summary.provenance is not None:
        with origin:
            provenance_badge(summary.provenance)
    with risk:
        assessment = summary.assessment
        if assessment is None:
            risk_chip(None, None)
        elif assessment.level is None:
            risk_chip(None, None, missing=text.unrated(assessment))
        else:
            risk_chip(assessment.level, assessment.decision.action)
    cdms.markdown(str(summary.cdm_count))
    if summary.active_job is not None:
        job.markdown(summary.active_job.value)
    with plan:
        st.page_link(
            PLANNER_PAGE,
            label="Planner",
            query_params={EVENT_PARAM: summary.event_id},
        )


def _reset_offset() -> None:
    """Back to the first page (the list has changed)."""
    st.session_state[OFFSET_KEY] = 0


def _move(step: int) -> None:
    """``step`` pages on (back if negative)."""
    st.session_state[OFFSET_KEY] = max(
        0, int(st.session_state.get(OFFSET_KEY, 0)) + step * PAGE_SIZE
    )


def _pager(page: SummaryPage) -> None:
    """Where the page is in the list, and buttons to the pages around it."""
    first = page.offset + 1
    last = page.offset + len(page.items)
    st.caption(f"Events {first} to {last} of {page.total}")
    back, forward, _ = st.columns([1, 1, 6])
    back.button(
        "Previous",
        disabled=page.offset == 0,
        on_click=_move,
        args=(-1,),
        key="events-previous",
    )
    forward.button(
        "Next",
        disabled=last >= page.total,
        on_click=_move,
        args=(1,),
        key="events-next",
    )


def _events(ctx: DashboardContext) -> None:
    """The list of events, a page at a time; the past ones on demand."""
    include_past = st.toggle("Show past events", key=PAST_KEY, on_change=_reset_offset)
    try:
        page = _read_page(ctx, include_past=include_past)
    except KeepoutError as exc:
        show_error(exc)
        return
    if not page.items:
        st.info("No events in the store." if include_past else "No upcoming events.")
        return
    _header()
    for summary in page.items:
        _row(summary, ctx.as_of)
    _pager(page)


def main() -> None:
    """The events page."""
    ctx = page_context()
    st.title("Events")
    _upload(ctx)
    _events(ctx)


main()
