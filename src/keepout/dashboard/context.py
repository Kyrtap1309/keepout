"""What the dashboard is run on: the store, the operator's settings and the
time it sees the data at, from the environment ``keepout dashboard`` sets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from astropy.time import Time

from keepout._environment import (
    AS_OF_VARIABLE,
    SETTINGS_VARIABLE,
    STORE_VARIABLE,
    parse_time,
)
from keepout.errors import StoreError
from keepout.settings import Settings, load_settings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from keepout.store import Store

__all__ = [
    "DashboardContext",
    "context_from_env",
    "open_existing",
    "parse_as_of",
]


def open_existing(store_path: Path) -> Store:
    """The dashboard's store, opened as every page opens it: never created.

    Use it as a context manager, which closes it.

    Raises
    ------
    StoreError
        If the store cannot be opened (:func:`keepout.store.open_store`
        with ``create=False``).
    """
    from keepout.store import open_store

    return open_store(store_path, create=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class DashboardContext:
    """What every page of the dashboard is run on.

    Parameters
    ----------
    store_path
        The store's file.
    settings
        The operator's settings: how Pc are computed and assessed.
    as_of
        The time the data are seen at (UTC): events whose latest TCA is
        before it are past.
    as_of_fixed
        Whether ``as_of`` was given (:data:`AS_OF_VARIABLE`) rather than
        read from the clock.
    """

    store_path: Path
    settings: Settings
    as_of: Time
    as_of_fixed: bool

    def store(self) -> Store:
        """The store, opened (:func:`open_existing`); use it as a context
        manager, which closes it.

        Raises
        ------
        StoreError
            If it cannot be opened.
        """
        return open_existing(self.store_path)


def parse_as_of(text: str) -> Time:
    """A time given as :data:`AS_OF_VARIABLE`, read as ``keepout dashboard
    --as-of`` reads it (:func:`keepout._environment.parse_time`): ISO 8601
    in UTC, e.g. ``2026-10-07T12:00``.

    Raises
    ------
    ValueError
        If ``text`` is not such a time; the message names the variable.
    """
    try:
        return parse_time(text)
    except ValueError as exc:
        raise ValueError(f"{AS_OF_VARIABLE}: {exc}") from exc


def context_from_env(
    env: Mapping[str, str] = os.environ, *, now: Callable[[], Time] = Time.now
) -> DashboardContext:
    """The dashboard's context from environment variables.

    Parameters
    ----------
    env
        The environment: :data:`STORE_VARIABLE` (required),
        :data:`SETTINGS_VARIABLE` and :data:`AS_OF_VARIABLE`; an empty value
        is as unset.
    now
        The clock, read when :data:`AS_OF_VARIABLE` is unset.

    Returns
    -------
    DashboardContext
        The store, the settings (the library's defaults without a file) and
        the time.

    Raises
    ------
    StoreError
        If :data:`STORE_VARIABLE` is unset. The store itself is not opened.
    SettingsError
        If the settings file is not a settings file; the message names the
        key.
    OSError
        If the settings file cannot be read.
    ValueError
        If :data:`AS_OF_VARIABLE` is not an ISO 8601 UTC time.
    """
    store = env.get(STORE_VARIABLE)
    if not store:
        raise StoreError(
            f"{STORE_VARIABLE} is not set: start the dashboard with "
            "keepout dashboard --store FILE"
        )
    settings_file = env.get(SETTINGS_VARIABLE)
    settings = load_settings(Path(settings_file)) if settings_file else Settings()
    as_of = env.get(AS_OF_VARIABLE)
    return DashboardContext(
        store_path=Path(store),
        settings=settings,
        as_of=parse_as_of(as_of) if as_of else now(),
        as_of_fixed=bool(as_of),
    )
