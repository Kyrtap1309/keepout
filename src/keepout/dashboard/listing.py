"""The events list's data, without Streamlit: a page of the store's events
as the events page reads it (``tools/dashboard_benchmark.py`` times it)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from keepout.dashboard.context import open_existing

if TYPE_CHECKING:
    from pathlib import Path

    from astropy.time import Time

    from keepout.settings import Settings
    from keepout.store import SummaryPage

__all__ = ["PAGE_SIZE", "read_page"]

PAGE_SIZE: Final = 25
"""Events on a page of the list."""


def read_page(
    store_path: Path,
    settings: Settings,
    *,
    as_of: Time,
    include_past: bool = False,
    offset: int = 0,
) -> SummaryPage:
    """The page of the list at ``offset``, read in one store opening, moved
    back to the last page if the list has become shorter.

    Parameters
    ----------
    store_path
        The store's file.
    settings
        The settings of the cached Pc and their thresholds.
    as_of
        The time of the list: events whose latest TCA is before it are past.
    include_past
        List the past events too.
    offset
        How many events of the list come before the page wanted.

    Returns
    -------
    SummaryPage
        The page; its ``offset`` is where it was read.

    Raises
    ------
    StoreError
        If the store cannot be opened or read.
    """
    with open_existing(store_path) as store:

        def read(start: int) -> SummaryPage:
            return store.summaries(
                settings,
                as_of=as_of,
                include_past=include_past,
                offset=start,
                limit=PAGE_SIZE,
            )

        page = read(offset)
        if not page.items and page.total and offset:
            page = read((page.total - 1) // PAGE_SIZE * PAGE_SIZE)
    return page
