"""The events page: the stored events, a page at a time, with their risk, and
the upload of CDM files."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import streamlit as st

from keepout._environment import MAX_UPLOAD_MB
from keepout._text import pc_cell, plural, utc
from keepout.dashboard import text
from keepout.dashboard.listing import PAGE_SIZE, page_of
from keepout.dashboard.routes import (
    OFFSET_KEY,
    PAST_KEY,
    UPLOAD_KEY,
    event_url,
    planner_url,
)
from keepout.dashboard.upload import MAX_FILES, UploadResult, ingest_upload
from keepout.dashboard.widgets import (
    job_status,
    page_context,
    read_jobs,
    shared_store,
    show_error,
    state,
)
from keepout.errors import KeepoutError
from keepout.figures import PROVENANCE_LABELS
from keepout.units import elapsed_seconds

if TYPE_CHECKING:
    from astropy.time import Time

    from keepout.dashboard.context import DashboardContext
    from keepout.store import EventSummary, IngestReport, SummaryPage


def _upload_message(report: IngestReport, jobs: int) -> str:
    """What an upload added and queued, in words."""
    return (
        f"Added {plural(len(report.added), 'CDM')} "
        f"({len(report.already_stored)} already stored) in "
        f"{plural(len(report.events), 'event')}; queued "
        f"{plural(jobs, 'Pc refresh job')}."
    )


def _ingest(ctx: DashboardContext, files: list[tuple[str, bytes]]) -> None:
    """Add uploaded files to the store, the outcome kept for
    :func:`_last_upload`; a store error is shown."""
    if not files:
        st.warning("Choose CDM files to add.")
        return
    try:
        result = ingest_upload(shared_store(ctx.store_path), ctx.settings, files)
    except KeepoutError as exc:
        show_error(exc)
        return
    st.session_state[UPLOAD_KEY] = result


def _last_upload(ctx: DashboardContext) -> None:
    """The last upload's errors or outcome and jobs, kept while a job is active."""
    result = state(UPLOAD_KEY, UploadResult)
    if result is None:
        return
    for error in result.errors:
        shown = text.literal(error.message)
        st.error(
            shown if error.file is None else f"{text.literal(error.file)}: {shown}"
        )
    if result.report is not None:
        st.success(_upload_message(result.report, len(result.jobs)))
    active = False
    if result.jobs:
        try:
            jobs = read_jobs(ctx.store_path, result.jobs)
        except KeepoutError as exc:
            show_error(exc)
        else:
            active = job_status(ctx.store_path, jobs)
    if not active:
        st.session_state.pop(UPLOAD_KEY, None)


def _upload(ctx: DashboardContext) -> None:
    """The upload form, collapsed: CDM files added to the store, their
    events' missing Pc queued; then the last upload's outcome."""
    with (
        st.expander("Upload CDMs", icon=":material/upload_file:"),
        st.form("upload", clear_on_submit=True),
    ):
        uploaded = st.file_uploader(
            f"CDM files: KVN or XML, plain or gzip-compressed, at most {MAX_FILES}",
            accept_multiple_files=True,
            max_upload_size=MAX_UPLOAD_MB,
        )
        submitted = st.form_submit_button("Add to store")
    if submitted:
        # An UploadedFile's name is typed as BinaryIO's (str | Any);
        # Streamlit sets it from the upload's name, a str.
        files = [(cast("str", file.name), file.getvalue()) for file in uploaded or []]
        _ingest(ctx, files)
    _last_upload(ctx)


def _read_page(ctx: DashboardContext, *, include_past: bool) -> SummaryPage:
    """The page of the list at the session's offset, moved back if shorter."""
    page = page_of(
        shared_store(ctx.store_path),
        ctx.settings,
        as_of=ctx.as_of,
        include_past=include_past,
        offset=state(OFFSET_KEY, int) or 0,
    )
    st.session_state[OFFSET_KEY] = page.offset
    return page


def _row(summary: EventSummary, as_of: Time, *, past: bool) -> dict[str, object]:
    """An event of the list; an upcoming one links to its planner too."""
    origin = summary.provenance
    job = summary.active_job
    row: dict[str, object] = {
        "Event": event_url(summary.event_id),
        "Objects": f"{summary.object1} / {summary.object2}",
        "TCA (UTC)": utc(summary.tca),
        "Until TCA": text.countdown(summary.tca, as_of),
        "Latest Pc": pc_cell(summary.latest, missing=text.NOT_COMPUTED),
        "Provenance": None if origin is None else PROVENANCE_LABELS[origin],
        "Risk": text.assessment_text(summary.assessment),
        "CDMs": summary.cdm_count,
        "Job": None if job is None else text.british(job.value),
    }
    if not past:
        row["Manoeuvre"] = planner_url(summary.event_id)
    return row


_COLUMNS = {
    "Event": st.column_config.LinkColumn(
        display_text=r"event=(.*)$", pinned=True, help="The event's page"
    ),
    "Latest Pc": st.column_config.TextColumn(alignment="right"),
    "CDMs": st.column_config.NumberColumn(format="%d"),
    "Manoeuvre": st.column_config.LinkColumn(
        display_text="Planner", help="The event's manoeuvre planner"
    ),
}


def _table(rows: list[dict[str, object]], key: str) -> None:
    """A table of events, sortable by any column."""
    st.dataframe(
        rows, column_config=_COLUMNS, hide_index=True, placeholder=text.MISSING, key=key
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
    buttons = st.container(horizontal=True)
    buttons.button(
        "Previous",
        disabled=page.offset == 0,
        on_click=_move,
        args=(-1,),
        key="events-previous",
    )
    buttons.button(
        "Next",
        disabled=last >= page.total,
        on_click=_move,
        args=(1,),
        key="events-next",
    )


def _events(ctx: DashboardContext) -> None:
    """The list of events, a page at a time; the past ones on demand, apart
    and without a planner."""
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
    upcoming = [i for i in page.items if elapsed_seconds(ctx.as_of, i.tca) >= 0]
    past = [i for i in page.items if elapsed_seconds(ctx.as_of, i.tca) < 0]
    if upcoming:
        _table([_row(item, ctx.as_of, past=False) for item in upcoming], "upcoming")
    if past:
        st.markdown("**Past events**")
        st.caption("Their TCA has gone by: nothing left to plan.")
        _table([_row(item, ctx.as_of, past=True) for item in past], "past")
    _pager(page)


def main() -> None:
    """The events page."""
    ctx = page_context()
    st.title("Events")
    _upload(ctx)
    _events(ctx)


main()
