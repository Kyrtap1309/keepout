"""A CDM built from a conjunction: the message keepout itself issues.

:func:`make_cdm` writes a :class:`~keepout.Conjunction` and what a CDM says
about each object (:class:`ObjectMetadata`: name, designators, type,
manoeuvrability, covariance method, comments) as a
:class:`~keepout.cdm.model.Cdm`: header, relative metadata with the miss
vector and relative velocity in the primary's RTN frame, and one section
per object with its GCRF state at TCA and its covariance in its own RTN
frame. The message reads back to the same conjunction:
``make_cdm(conjunction, ...).conjunction()`` gives the input's objects,
states, covariances and hard-body radius, also after a KVN or XML
round-trip.

The builder knows nothing of where the covariances come from; a synthetic
operator says so in ``COVARIANCE_METHOD = DEFAULT`` and the covariance
comments of :class:`ObjectMetadata`.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from astropy.time import Time

from keepout._validation import is_single_line
from keepout.ccsds.message import COMMENT
from keepout.cdm.keywords import (
    STATE_KEYWORDS,
    VERSION_1_0,
    VERSION_2_0,
    VERSION_KEYWORD,
    covariance_entries,
)
from keepout.cdm.model import Cdm, CovarianceMethod, Field, Section
from keepout.conjunction import Conjunction
from keepout.frames import rtn_rotation
from keepout.objects import ObjectState

__all__ = [
    "MESSAGE_ID_PRECISION",
    "ORIGINATOR",
    "SCREEN_VOLUME_COMMENT",
    "TCA_PRECISION",
    "CovarianceMethod",
    "Maneuverable",
    "ObjectMetadata",
    "ObjectType",
    "RtnSemiAxes",
    "make_cdm",
    "make_message_id",
]

ORIGINATOR: Final = "KEEPOUT"
"""Default ``ORIGINATOR`` of the messages keepout builds."""

TCA_PRECISION: Final = 6
"""Fractional digits of the seconds of ``TCA`` (a microsecond: the states
are at TCA well within :data:`keepout.objects.EPOCH_TOLERANCE`)."""

MESSAGE_ID_PRECISION: Final = 3
"""Fractional digits of the seconds of the times in a message id."""

SCREEN_VOLUME_COMMENT: Final = (
    "SCREEN_VOLUME_X, _Y and _Z are the semi-axes of the RTN ellipsoid"
)
"""Relative-metadata comment of a message with a screening volume: CCSDS
508.0-B-1 calls them the volume's "component size", which leaves open
whether they are semi-axes or full extents."""


class ObjectType(enum.StrEnum):
    """``OBJECT_TYPE`` values of CCSDS 508.0-B-1."""

    PAYLOAD = "PAYLOAD"
    """A payload."""
    ROCKET_BODY = "ROCKET BODY"
    """A rocket body."""
    DEBRIS = "DEBRIS"
    """Debris."""
    UNKNOWN = "UNKNOWN"
    """Type not known."""
    OTHER = "OTHER"
    """None of the above."""


class Maneuverable(enum.StrEnum):
    """``MANEUVERABLE`` values of CCSDS 508.0-B-1."""

    YES = "YES"
    """The object can manoeuvre."""
    NO = "NO"
    """The object cannot manoeuvre."""
    NOT_APPLICABLE = "N/A"
    """Not applicable or not known (e.g. a catalog object)."""


def _check_comments(comments: Sequence[str], what: str) -> tuple[str, ...]:
    """``comments`` as a tuple; each must survive a KVN and XML round-trip.

    Raises
    ------
    ValueError
        If a comment has more than one line or leading or trailing blanks.
    """
    for text in comments:
        if not is_single_line(text):
            raise ValueError(
                f"{what} comment {text!r} must be one line without leading or "
                "trailing blanks"
            )
    return tuple(comments)


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    """What a CDM says about one object besides its state and covariance.

    The object's identifier (``OBJECT_DESIGNATOR``), state, covariance and
    hard-body radius come from its :class:`~keepout.objects.ObjectState`.

    Parameters
    ----------
    name
        ``OBJECT_NAME``.
    international_designator
        ``INTERNATIONAL_DESIGNATOR`` (COSPAR id, e.g. ``1998-067A``; CCSDS
        allows ``UNKNOWN``).
    catalog_name
        ``CATALOG_NAME``: the catalog the designator belongs to.
    object_type
        ``OBJECT_TYPE``; not written if ``None``.
    maneuverable
        ``MANEUVERABLE``; ``N/A`` (not known) by default.
    covariance_method
        ``COVARIANCE_METHOD``: ``DEFAULT`` for an assumed covariance.
    ephemeris_name
        ``EPHEMERIS_NAME``: ``NONE`` when no operator ephemeris was used.
    comments
        Comments opening the object section.
    covariance_comments
        Comments opening the covariance block (e.g. the covariance model).

    Raises
    ------
    ValueError
        If an enumerated field has a value CCSDS does not define, or a
        comment has more than one line or leading or trailing blanks.
    """

    name: str
    international_designator: str
    catalog_name: str = "SATCAT"
    object_type: ObjectType | None = None
    maneuverable: Maneuverable = Maneuverable.NOT_APPLICABLE
    covariance_method: CovarianceMethod = CovarianceMethod.CALCULATED
    ephemeris_name: str = "NONE"
    comments: tuple[str, ...] = ()
    covariance_comments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.object_type is not None:
            object.__setattr__(self, "object_type", ObjectType(self.object_type))
        object.__setattr__(self, "maneuverable", Maneuverable(self.maneuverable))
        object.__setattr__(
            self, "covariance_method", CovarianceMethod(self.covariance_method)
        )
        object.__setattr__(self, "comments", _check_comments(self.comments, "object"))
        object.__setattr__(
            self,
            "covariance_comments",
            _check_comments(self.covariance_comments, "covariance"),
        )


class RtnSemiAxes(Protocol):
    """Semi-axes of an RTN ellipsoid [m], such as a
    :class:`keepout.screening.ScreeningVolume`."""

    @property
    def radial(self) -> float:
        """Semi-axis along R [m]."""
        ...

    @property
    def transverse(self) -> float:
        """Semi-axis along T [m]."""
        ...

    @property
    def normal(self) -> float:
        """Semi-axis along N [m]."""
        ...


def _with_precision(time: Time, precision: int) -> Time:
    """A UTC copy of ``time`` written with ``precision`` fractional digits."""
    copy = Time(time.utc, copy=True)
    copy.precision = precision
    return copy


def _stamp(time: Time) -> str:
    """``time`` in UTC to the millisecond, e.g. ``20261008T123456.789Z``."""
    text = str(_with_precision(time, MESSAGE_ID_PRECISION).isot)
    return text.replace("-", "").replace(":", "") + "Z"


def make_message_id(
    primary_id: str, secondary_id: str, tca: Time, created: Time
) -> str:
    """The deterministic ``MESSAGE_ID`` of a message about a conjunction.

    Both identifiers, the TCA and the creation time, each time in UTC to
    the millisecond, e.g.
    ``25544_49863_20261008T123456.789Z_20261006T000000.000Z``. The same
    message built again gets the same id, and the messages about one
    approach issued at different times (successive CDMs of one event)
    get different ids. The id is also a portable file name for
    identifiers of letters, digits, ``-`` and ``.``.

    Parameters
    ----------
    primary_id, secondary_id
        Identifiers of the two objects (e.g. catalog numbers).
    tca
        Time of closest approach.
    created
        Creation time of the message (its ``CREATION_DATE``).
    """
    return f"{primary_id}_{secondary_id}_{_stamp(tca)}_{_stamp(created)}"


def _hard_body_radii(conjunction: Conjunction) -> tuple[float, float] | None:
    """Each object's ``HBR``, or None if neither object has a radius.

    Raises
    ------
    ValueError
        If only one object has a radius, or the two do not add up to the
        conjunction's combined radius: a CDM gives the radius per object.
    """
    primary = conjunction.primary.hard_body_radius
    secondary = conjunction.secondary.hard_body_radius
    if primary is None and secondary is None:
        return None
    combined = conjunction.combined_hard_body_radius
    if (
        primary is None
        or secondary is None
        or not math.isclose(primary + secondary, combined)
    ):
        raise ValueError(
            "a CDM gives the hard-body radius per object (HBR): the objects' "
            f"radii {primary} and {secondary} must add up to the combined radius "
            f"{combined} m, or both be None"
        )
    return primary, secondary


def _relative(conjunction: Conjunction) -> list[Field]:
    """``MISS_DISTANCE`` to ``RELATIVE_VELOCITY_N``, in the primary's RTN."""
    primary = conjunction.primary.state
    rotation = rtn_rotation(primary.r, primary.v)
    position = rotation @ conjunction.relative_position
    velocity = rotation @ conjunction.relative_velocity
    return [
        Field("MISS_DISTANCE", float(np.linalg.norm(conjunction.relative_position))),
        Field("RELATIVE_SPEED", conjunction.relative_speed),
        *(
            Field(f"RELATIVE_POSITION_{axis}", float(value))
            for axis, value in zip("RTN", position, strict=True)
        ),
        *(
            Field(f"RELATIVE_VELOCITY_{axis}", float(value))
            for axis, value in zip("RTN", velocity, strict=True)
        ),
    ]


def _screening(
    period: tuple[Time, Time] | None, volume: RtnSemiAxes | None
) -> list[Field]:
    fields: list[Field] = []
    if period is not None:
        start, stop = period
        if not stop > start:
            raise ValueError("the screening period must end after it starts")
        fields += [
            Field("START_SCREEN_PERIOD", start),
            Field("STOP_SCREEN_PERIOD", stop),
        ]
    if volume is not None:
        fields += [
            Field("SCREEN_VOLUME_FRAME", "RTN"),
            Field("SCREEN_VOLUME_SHAPE", "ELLIPSOID"),
            Field("SCREEN_VOLUME_X", float(volume.radial)),
            Field("SCREEN_VOLUME_Y", float(volume.transverse)),
            Field("SCREEN_VOLUME_Z", float(volume.normal)),
        ]
    return fields


def _object_section(
    index: int,
    item: ObjectState,
    metadata: ObjectMetadata,
    hard_body_radius: float | None,
) -> Section:
    """Object section ``index`` in the field order of the CCSDS schema."""
    fields = [Field(COMMENT, text) for text in metadata.comments]
    fields += [
        Field("OBJECT", f"OBJECT{index}"),
        Field("OBJECT_DESIGNATOR", item.object_id),
        Field("CATALOG_NAME", metadata.catalog_name),
        Field("OBJECT_NAME", metadata.name),
        Field("INTERNATIONAL_DESIGNATOR", metadata.international_designator),
    ]
    if metadata.object_type is not None:
        fields.append(Field("OBJECT_TYPE", str(metadata.object_type)))
    fields += [
        Field("EPHEMERIS_NAME", metadata.ephemeris_name),
        Field("COVARIANCE_METHOD", str(metadata.covariance_method)),
        Field("MANEUVERABLE", str(metadata.maneuverable)),
        Field("REF_FRAME", item.state.frame),
    ]
    if hard_body_radius is not None:
        fields.append(Field("HBR", hard_body_radius))
    state = item.state
    vector = (*state.r, *state.v)
    fields += [
        Field(name, float(value))
        for name, value in zip(STATE_KEYWORDS, vector, strict=True)
    ]
    fields += [Field(COMMENT, text) for text in metadata.covariance_comments]
    # An unknown covariance is written as zeros, as operators do.
    matrix = (
        np.zeros((6, 6))
        if item.covariance is None
        else item.covariance.in_rtn(state).matrix
    )
    fields += [
        Field(name, float(matrix[row, column]))
        for name, row, column in covariance_entries(6)
    ]
    return Section("segment", tuple(fields))


def make_cdm(
    conjunction: Conjunction,
    primary_metadata: ObjectMetadata,
    secondary_metadata: ObjectMetadata,
    *,
    created: Time,
    originator: str = ORIGINATOR,
    message_id: str | None = None,
    message_for: str | None = None,
    screening_period: tuple[Time, Time] | None = None,
    screening_volume: RtnSemiAxes | None = None,
    comments: Sequence[str] = (),
) -> Cdm:
    """The CDM of a conjunction (``OBJECT1`` is the primary).

    Each object's ``OBJECT_DESIGNATOR`` is its ``object_id``; its state is
    written in GCRF (``REF_FRAME = GCRF``) and its covariance in its own
    RTN frame, all zeros if unknown. ``MISS_DISTANCE`` is the range at TCA,
    ``RELATIVE_POSITION_*`` and ``RELATIVE_VELOCITY_*`` are the secondary's
    position and velocity relative to the primary projected on the
    primary's RTN axes. The message is CCSDS 508.0-B-1 (version 1.0) unless
    both objects have a hard-body radius: ``HBR`` is a CDM 2.0 keyword, so
    the message is then version 2.0. The result is deterministic: the same
    inputs give the same message.

    Parameters
    ----------
    conjunction
        The conjunction; ``conjunction.tca`` is the ``TCA``, written to the
        microsecond (:data:`TCA_PRECISION`).
    primary_metadata, secondary_metadata
        What the message says about each object.
    created
        ``CREATION_DATE``, as given (no clock is read, so that a message
        can be built again identically).
    originator
        ``ORIGINATOR``.
    message_id
        ``MESSAGE_ID``; by default :func:`make_message_id` of the objects'
        ids, the TCA and ``created``.
    message_for
        ``MESSAGE_FOR`` (the name of the spacecraft the message is for), if
        any.
    screening_period
        ``START_SCREEN_PERIOD`` and ``STOP_SCREEN_PERIOD``, if given.
    screening_volume
        The screening ellipsoid's RTN semi-axes [m], if given: written as
        ``SCREEN_VOLUME_X``, ``_Y`` and ``_Z``, with ``SCREEN_VOLUME_FRAME``
        ``RTN``, ``SCREEN_VOLUME_SHAPE`` ``ELLIPSOID`` and the comment
        :data:`SCREEN_VOLUME_COMMENT`.
    comments
        Header comments.

    Returns
    -------
    Cdm
        The message; ``.conjunction()`` gives ``conjunction`` back.

    Raises
    ------
    ValueError
        If only one object has a hard-body radius or the two do not add up
        to the combined radius, the screening period does not end after it
        starts, a comment has more than one line or leading or trailing
        blanks, or a text field is empty.
    """
    radii = _hard_body_radii(conjunction)
    tca = _with_precision(conjunction.tca, TCA_PRECISION)
    if message_id is None:
        message_id = make_message_id(
            conjunction.primary.object_id,
            conjunction.secondary.object_id,
            tca,
            created,
        )
    header = [
        Field(VERSION_KEYWORD, VERSION_1_0 if radii is None else VERSION_2_0),
        *(Field(COMMENT, text) for text in _check_comments(comments, "header")),
        Field("CREATION_DATE", created),
        Field("ORIGINATOR", originator),
    ]
    if message_for is not None:
        header.append(Field("MESSAGE_FOR", message_for))
    header.append(Field("MESSAGE_ID", message_id))
    relative = (
        [] if screening_volume is None else [Field(COMMENT, SCREEN_VOLUME_COMMENT)]
    )
    relative += [
        Field("TCA", tca),
        *_relative(conjunction),
        *_screening(screening_period, screening_volume),
    ]
    primary_radius, secondary_radius = (None, None) if radii is None else radii
    return Cdm(
        Section("header", tuple(header)),
        Section("relative", tuple(relative)),
        (
            _object_section(1, conjunction.primary, primary_metadata, primary_radius),
            _object_section(
                2, conjunction.secondary, secondary_metadata, secondary_radius
            ),
        ),
    )
