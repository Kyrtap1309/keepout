"""The JSON of a cached observation, and the key it is cached under.

A store keeps each CDM's :class:`~keepout.risk.PcObservation` as JSON
(:func:`observation_to_json`, :func:`observation_from_json`): every field but
:attr:`~keepout.risk.PcObservation.result`, whose arrays and details a store
need not keep; :attr:`~keepout.risk.PcObservation.method` still names the
method. The key it is cached under (:func:`result_key`) is a hash of all that
changes a Pc: keepout's version, the method and its options, the covariance
policy and the hard-body radius. Thresholds are not in it: they are applied
when the observations are read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, TypeVar

import keepout
from keepout._validation import require_choice
from keepout.errors import StoreError
from keepout.risk.observation import PC_KINDS, PcObservation

if TYPE_CHECKING:
    from keepout._types import JsonValue
    from keepout.covariance import Policy
    from keepout.pc import PcComputation

__all__ = [
    "OBSERVATION_KEYS",
    "observation_from_json",
    "observation_to_json",
    "result_key",
]

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
        The observation.
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
    if isinstance(value, bool) or not isinstance(value, int | float):
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
    diagnostics = data["diagnostics"]
    if not isinstance(diagnostics, list) or not all(
        isinstance(item, str) for item in diagnostics
    ):
        raise StoreError(
            f"{_WHERE}diagnostics must be a list of strings, got {diagnostics!r}"
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


def result_key(
    *,
    compute: PcComputation,
    policy: Policy,
    combined_hard_body_radius: float | None,
) -> str:
    """The key of the observations computed with these parameters: the
    SHA-256 (hex) of their JSON and keepout's version.

    A new keepout version, method, option, policy or radius gives a new key,
    so a store never serves a Pc computed otherwise. A radius keys the same
    as an integer or a float (``20`` and ``20.0``).

    Parameters
    ----------
    compute
        The Pc method and its options.
    policy
        What is done with an invalid covariance.
    combined_hard_body_radius
        The combined hard-body radius [m], or ``None`` for the CDMs'.
    """
    text = json.dumps(
        {
            "keepout": keepout.__version__,
            "method": compute.method,
            "options": sorted(compute.options.items()),
            "policy": policy,
            "hbr": None
            if combined_hard_body_radius is None
            else float(combined_hard_body_radius),
        },
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
