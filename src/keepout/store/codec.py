"""The JSON of a cached observation, and the keys of the store's cache.

A store keeps each CDM's :class:`~keepout.risk.PcObservation` as JSON
(:func:`observation_to_json`, :func:`observation_from_json`): every field but
:attr:`~keepout.risk.PcObservation.result`, whose arrays and details a store
need not keep; :attr:`~keepout.risk.PcObservation.method` still names the
method. The key it is cached under (:func:`result_key`) is a hash of all that
changes a Pc: keepout's version, the method and its options, the covariance
policy and the hard-body radius. Thresholds are not in it: they are applied
when the observations are read.

A manoeuvre outcome is cached under :func:`maneuver_key`. It hashes all that
changes a plan: keepout's version, the operator's settings and the digests
of the catalog's files. Thresholds and target are among the settings here.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final, TypeVar, cast

import keepout
from keepout._validation import is_number, require_choice
from keepout.errors import StoreError
from keepout.risk.observation import PC_KINDS, PcObservation
from keepout.settings import settings_to_data

if TYPE_CHECKING:
    from collections.abc import Mapping

    from keepout._types import JsonValue
    from keepout.settings import Settings
    from keepout.sources.catalog import CatalogReference

__all__ = [
    "MANEUVER_KIND",
    "OBSERVATION_KEYS",
    "maneuver_key",
    "observation_from_json",
    "observation_to_json",
    "result_key",
]

MANEUVER_KIND: Final = "maneuver"
"""The ``kind`` in the text a manoeuvre key hashes: no key of a result has
one."""

OBSERVATION_KEYS: Final = frozenset(
    {
        "time_to_tca",
        "source",
        "pc",
        "kind",
        "ci_high",
        "error",
        "message_id",
        "reported_pc",
        "diagnostics",
        "assumed_covariance",
        "method",
    }
)
"""The keys of an observation's JSON; all required."""

_WHERE: Final = "cached observation: "

_T = TypeVar("_T")


def observation_to_json(observation: PcObservation) -> dict[str, JsonValue]:
    """The JSON of an observation, without its full result.

    Parameters
    ----------
    observation
        The observation, e.g. of :meth:`~keepout.events.ConjunctionEvent.observations`.

    Returns
    -------
    dict[str, JsonValue]
        The JSON object.
    """
    return {
        "time_to_tca": observation.time_to_tca,
        "source": observation.source,
        "pc": observation.pc,
        "kind": observation.kind,
        "ci_high": observation.ci_high,
        "error": observation.error,
        "message_id": observation.message_id,
        "reported_pc": observation.reported_pc,
        "diagnostics": list(observation.diagnostics),
        "assumed_covariance": observation.assumed_covariance,
        "method": observation.method,
    }


def _number(data: Mapping[str, object], key: str) -> float | None:
    """``data[key]``, a number or ``None``.

    Raises
    ------
    StoreError
        If it is neither (a bool is not a number).
    """
    value = data[key]
    if value is None:
        return None
    if not is_number(value):
        raise StoreError(f"{_WHERE}{key} must be a number, got {value!r}")
    return float(value)


def _text(data: Mapping[str, object], key: str) -> str | None:
    """``data[key]``, a string or ``None``.

    Raises
    ------
    StoreError
        If it is neither.
    """
    value = data[key]
    if value is not None and not isinstance(value, str):
        raise StoreError(f"{_WHERE}{key} must be a string, got {value!r}")
    return value


def _required(value: _T | None, key: str) -> _T:
    """``value``, the value of ``key``, checked to be given.

    Raises
    ------
    StoreError
        If it is ``None``.
    """
    if value is None:
        raise StoreError(f"{_WHERE}{key} must not be null")
    return value


def observation_from_json(data: Mapping[str, object]) -> PcObservation:
    """The observation of its JSON (:func:`observation_to_json`), without a
    full result.

    Parameters
    ----------
    data
        The JSON, as :func:`json.loads` returns it.

    Returns
    -------
    PcObservation
        The observation.

    Raises
    ------
    StoreError
        If a key is unknown or missing, a value has the wrong type, or the
        values make no valid observation; the message names the key.
    """
    unknown = sorted(set(data) - OBSERVATION_KEYS)
    if unknown:
        raise StoreError(f"{_WHERE}unknown key(s) {', '.join(unknown)}")
    missing = sorted(OBSERVATION_KEYS - set(data))
    if missing:
        raise StoreError(f"{_WHERE}missing key(s) {', '.join(missing)}")
    listed = data["diagnostics"]
    items = cast("list[object]", listed) if isinstance(listed, list) else None
    diagnostics = [item for item in items or () if isinstance(item, str)]
    if items is None or len(diagnostics) != len(items):
        raise StoreError(
            f"{_WHERE}diagnostics must be a list of strings, got {listed!r}"
        )
    assumed = data["assumed_covariance"]
    if not isinstance(assumed, bool):
        raise StoreError(
            f"{_WHERE}assumed_covariance must be true or false, got {assumed!r}"
        )
    try:
        return PcObservation(
            time_to_tca=_required(_number(data, "time_to_tca"), "time_to_tca"),
            source=_required(_text(data, "source"), "source"),
            pc=_number(data, "pc"),
            kind=require_choice(data["kind"], PC_KINDS, "kind"),
            ci_high=_number(data, "ci_high"),
            error=_text(data, "error"),
            message_id=_text(data, "message_id"),
            reported_pc=_number(data, "reported_pc"),
            diagnostics=tuple(diagnostics),
            assumed_covariance=assumed,
            method=_text(data, "method"),
        )
    except ValueError as exc:
        raise StoreError(f"{_WHERE}{exc}") from exc


def _key(fields: Mapping[str, object]) -> str:
    """The SHA-256 (hex) of the JSON of keepout's version and ``fields``,
    keys sorted."""
    text = json.dumps(
        {"keepout": keepout.__version__, **fields}, sort_keys=True, default=repr
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def result_key(settings: Settings) -> str:
    """The key of the observations ``settings`` compute: the SHA-256 (hex)
    of the JSON of keepout's version and what in ``settings`` changes a Pc.

    A new keepout version, method, option, policy or radius gives a new key,
    so a store never serves a Pc computed otherwise; the thresholds and the
    manoeuvre target do not change it. A radius keys the same as an integer
    or a float (``20`` and ``20.0``).

    Parameters
    ----------
    settings
        The operator's settings.

    Returns
    -------
    str
        The key, lower-case hexadecimal.
    """
    compute, radius = settings.compute, settings.combined_hard_body_radius
    return _key(
        {
            "method": compute.method,
            "options": sorted(compute.options.items()),
            "policy": settings.policy,
            "hbr": None if radius is None else float(radius),
        }
    )


def maneuver_key(settings: Settings, catalog: CatalogReference | None) -> str:
    """The cache key of a plan: the SHA-256 (hex) of keepout's version, the
    job kind (:data:`MANEUVER_KIND`), the settings and the sorted digests of
    the catalog's files.

    The settings are hashed as a settings file holds them. The catalog's
    label and the files' names and paths do not change the key: the same
    bytes give the same re-screens.

    Parameters
    ----------
    settings
        The settings of the plan.
    catalog
        The catalog the burns are re-screened against; ``None``: none.

    Returns
    -------
    str
        The key, lower-case hexadecimal.

    Raises
    ------
    ValueError
        If a settings file cannot hold ``settings``, as
        :func:`~keepout.settings.settings_to_data`.
    """
    return _key(
        {
            "kind": MANEUVER_KIND,
            "settings": settings_to_data(settings),
            "catalog": None
            if catalog is None
            else sorted(file.sha256 for file in catalog.files),
        }
    )
