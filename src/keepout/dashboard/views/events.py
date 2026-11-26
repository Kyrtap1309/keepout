"""The events page: the stored events, a page at a time, with their risk, and
the upload of CDM files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import streamlit as st

from keepout._environment import MAX_UPLOAD_MB
from keepout._text import plural, utc
from keepout.dashboard import text
from keepout.dashboard.listing import PAGE_SIZE, read_page
from keepout.dashboard.routes import (
    EVENT_PAGE,
    EVENT_PARAM,
    OFFSET_KEY,
    PAST_KEY,
    PLANNER_PAGE,
    UPLOAD_KEY,
)
from keepout.dashboard.upload import MAX_FILES, UploadResult, ingest_upload
from keepout.dashboard.widgets import (
    assessment_chip,
    job_status,
    page_context,
    provenance_badge,
    show_error,
    state,
    table_header,
    table_row,
)
from keepout.errors import KeepoutError

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.dashboard.context import DashboardContext
    from keepout.store import EventSummary, IngestReport, SummaryPage

# (No docstring: Streamlit would show a bare string as Markdown.)
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
    st.session_state.pop(UPLOAD_KEY, None)
    if not files:
        st.warning("Choose CDM files to ingest.")
        return
    try:
        with ctx.store() as store:
            st.session_state[UPLOAD_KEY] = ingest_upload(store, ctx.settings, files)
    except KeepoutError as exc:
        show_error(exc)


def _last_upload(ctx: DashboardContext) -> None:
    """The outcome of the session's last upload: its errors, each naming its
    file, or what it stored and the status of the jobs it queued. It is
    shown once, and again only while a job it queued is queued or running."""
    result = state(UPLOAD_KEY, UploadResult)
    st.session_state.pop(UPLOAD_KEY, None)
    if result is None:
        return
    for error in result.errors:
        shown = text.literal(error.message)
        st.error(
            shown if error.file is None else f"{text.literal(error.file)}: {shown}"
        )
    if result.report is not None:
        st.success(_upload_message(result.report, len(result.jobs)))
    if result.jobs and job_status(ctx.store_path, result.jobs):
        st.session_state[UPLOAD_KEY] = result


def _upload(ctx: DashboardContext) -> None:
    """The upload form: CDM files ingested, their events' missing Pc queued;
    the outcome of the last upload (:func:`_last_upload`)."""
    with st.expander("Upload CDMs", expanded=True):
        with st.form("upload", clear_on_submit=True):
            uploaded = st.file_uploader(
                f"CDM files (KVN or XML, gzip-compressed or not; at most "
                f"{MAX_FILES} files of {MAX_UPLOAD_MB} MB each)",
                accept_multiple_files=True,
                max_upload_size=MAX_UPLOAD_MB,
            )
            submitted = st.form_submit_button("Ingest")
        if submitted:
            # An UploadedFile's name is typed as BinaryIO's (str | Any);
            # Streamlit sets it from the upload's name, a str.
            files = [
                (cast("str", file.name), file.getvalue()) for file in uploaded or []
            ]
            _ingest(ctx, files)
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
        offset=state(OFFSET_KEY, int) or 0,
    )
    st.session_state[OFFSET_KEY] = page.offset
    return page


def _row(summary: EventSummary, as_of: Time) -> None:
    """An event of the list, its id a link to its page; a link to its
    manoeuvre planner."""
    cells = table_row(COLUMNS)
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
    pc.markdown(text.pc_cell(summary.latest))
    if summary.provenance is not None:
        with origin:
            provenance_badge(summary.provenance)
    with risk:
        assessment_chip(summary.assessment)
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
        0, (state(OFFSET_KEY, int) or 0) + step * PAGE_SIZE
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
        st.info(
            "No events in the store: upload CDMs above."
            if include_past
            else f"No upcoming events as of {utc(ctx.as_of)} UTC: turn on Show "
            "past events, or start the dashboard with an earlier --as-of."
        )
        return
    table_header(COLUMNS)
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
