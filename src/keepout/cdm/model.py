"""In-memory Conjunction Data Message.

A :class:`Cdm` keeps every field in message order — comments and unknown
keywords included — so KVN and XML round-trips are lossless. Known keywords
hold typed SI values (see :mod:`keepout.cdm.keywords`); unknown keywords hold
their raw text and unit.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Final, assert_never

import numpy as np
from astropy.time import Time

from keepout._types import FloatArray, read_only
from keepout.ccsds.keywords import FieldValue, ValueKind, format_value, parse_value
from keepout.ccsds.message import COMMENT, USER_DEFINED_PREFIX
from keepout.cdm.keywords import (
    KEYWORDS,
    POSITION_KEYWORDS,
    REQUIRED,
    VELOCITY_KEYWORDS,
    VERSION_KEYWORD,
    SectionKind,
    covariance_matrix,
    covariance_names,
)
from keepout.conjunction import Conjunction, combined_hard_body_radius
from keepout.covariance import Covariance, Policy, make_covariance
from keepout.errors import CdmParseError
from keepout.objects import ObjectState, ReportedObject, gcrf_object_states
from keepout.state import StateVector, as_ref_frame

__all__ = [
    "COMMENT",
    "KEYWORD_PATTERN",
    "USER_DEFINED_PREFIX",
    "Cdm",
    "CdmObject",
    "CovarianceMethod",
    "DuplicateKeyword",
    "Field",
    "Section",
    "covariance_dimension",
    "format_field",
    "parse_field",
]


KEYWORD_PATTERN: Final = re.compile(r"[A-Z][A-Z0-9_]*")
"""What a keyword may look like: valid in KVN and as an XML element name."""

_COVARIANCE_NAMES: Final[Mapping[int, frozenset[str]]] = {
    size: frozenset(covariance_names(size)) for size in range(6, 10)
}


class CovarianceMethod(enum.StrEnum):
    """``COVARIANCE_METHOD`` values of CCSDS 508.0-B-1."""

    CALCULATED = "CALCULATED"
    """Computed in the orbit determination."""
    DEFAULT = "DEFAULT"
    """Assumed (a default or a model), not from an orbit determination."""


class DuplicateKeyword(ValueError):
    """A section repeats a keyword.

    Parameters
    ----------
    kind
        Section kind.
    keyword
        The repeated keyword.
    index
        Position of the repeat in the section's fields, so that readers can
        report where it is in the message.
    """

    def __init__(self, kind: SectionKind, keyword: str, index: int) -> None:
        super().__init__(f"{kind} section repeats keyword {keyword}")
        self.keyword = keyword
        self.index = index


@dataclass(frozen=True, slots=True)
class Field:
    """One ``KEYWORD = value`` entry (or a ``COMMENT``).

    Parameters
    ----------
    keyword
        CDM keyword, or ``"COMMENT"``.
    value
        SI value for known keywords; raw text for comments and unknown keywords.
    unit
        Raw unit text; only for unknown keywords (known units are fixed).
    """

    keyword: str
    value: FieldValue
    unit: str | None = None


def parse_field(kind: SectionKind, keyword: str, text: str, unit: str | None) -> Field:
    """A field from message text: typed SI value if known, raw text if not.

    Parameters
    ----------
    kind
        Section the field belongs to.
    keyword
        CDM keyword (not ``COMMENT``).
    text
        Value text, without the unit.
    unit
        Unit given in the message, if any.

    Raises
    ------
    ValueError
        If a known keyword has a wrong unit or an invalid value.
    """
    spec = KEYWORDS[kind].get(keyword)
    if spec is None:
        return Field(keyword, text, unit)
    return Field(keyword, parse_value(spec, text, unit))


def format_field(kind: SectionKind, field: Field) -> tuple[str, str | None]:
    """Message text and unit of a field (not ``COMMENT``).

    Parameters
    ----------
    kind
        Section the field belongs to.
    field
        Field to write.
    """
    spec = KEYWORDS[kind].get(field.keyword)
    if spec is None:
        return str(field.value), field.unit
    return format_value(spec, field.value)


def _matches(value: object, kind: ValueKind) -> bool:
    match kind:
        case "time":
            return isinstance(value, Time) and bool(value.isscalar)
        case "int":
            return isinstance(value, int) and not isinstance(value, bool)
        case "float":
            return isinstance(value, float)
        case "str":
            return isinstance(value, str)
        case _:
            assert_never(kind)


@dataclass(frozen=True)
class Section:
    """An ordered group of fields: header, relative metadata or one object.

    Parameters
    ----------
    kind
        ``"header"``, ``"relative"`` or ``"segment"``.
    fields
        Fields in message order.

    Raises
    ------
    DuplicateKeyword
        If a keyword is repeated.
    ValueError
        For an unknown kind, a malformed keyword, a value of the wrong type or
        a unit on a known keyword.
    """

    kind: SectionKind
    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        if self.kind not in KEYWORDS:
            raise ValueError(f"unknown section kind {self.kind!r}")
        table = KEYWORDS[self.kind]
        seen: set[str] = set()
        for index, field in enumerate(self.fields):
            if not KEYWORD_PATTERN.fullmatch(field.keyword):
                raise ValueError(
                    f"keyword {field.keyword!r} must be upper-case letters, digits "
                    "and underscores, starting with a letter"
                )
            if field.keyword != COMMENT:
                if field.keyword in seen:
                    raise DuplicateKeyword(self.kind, field.keyword, index)
                seen.add(field.keyword)
            spec = table.get(field.keyword)
            kind: ValueKind = "str" if spec is None else spec.kind
            if not _matches(field.value, kind):
                raise ValueError(
                    f"{field.keyword} value {field.value!r} is not of kind {kind!r}"
                )
            if spec is not None and field.unit is not None:
                raise ValueError(
                    f"{field.keyword}: unit is fixed by the standard; "
                    "store the SI value with unit=None"
                )

    @cached_property
    def _values(self) -> dict[str, FieldValue]:
        return {f.keyword: f.value for f in self.fields if f.keyword != COMMENT}

    def get(self, keyword: str) -> FieldValue | None:
        """Value of the field with this keyword, or None.

        Parameters
        ----------
        keyword
            Keyword to look up (not ``COMMENT``; see :attr:`comments`).
        """
        return self._values.get(keyword)

    @property
    def comments(self) -> tuple[str, ...]:
        """Comment texts in order."""
        return tuple(str(f.value) for f in self.fields if f.keyword == COMMENT)

    @property
    def extras(self) -> dict[str, str]:
        """Keywords not in the :data:`~keepout.cdm.keywords.KEYWORDS` table, as text."""
        table = KEYWORDS[self.kind]
        return {
            f.keyword: str(f.value)
            for f in self.fields
            if f.keyword != COMMENT and f.keyword not in table
        }

    def missing(self) -> tuple[str, ...]:
        """Required keywords absent from this section or given no value."""
        return tuple(k for k in REQUIRED[self.kind] if self.get(k) in (None, ""))


def _text(section: Section, keyword: str) -> str:
    value = section.get(keyword)
    if not isinstance(value, str):
        raise KeyError(keyword)
    return value


def _optional_text(section: Section, keyword: str) -> str | None:
    value = section.get(keyword)
    return value if isinstance(value, str) else None


def _optional_number(section: Section, keyword: str) -> float | None:
    value = section.get(keyword)
    return float(value) if isinstance(value, int | float) else None


def _number(section: Section, keyword: str) -> float:
    value = _optional_number(section, keyword)
    if value is None:
        raise KeyError(keyword)
    return value


def _epoch(section: Section, keyword: str) -> Time:
    value = section.get(keyword)
    if not isinstance(value, Time):
        raise KeyError(keyword)
    return value


def covariance_dimension(section: Section) -> int:
    """Size of the covariance block present in an object section.

    Parameters
    ----------
    section
        An object section.

    Raises
    ------
    ValueError
        If the entries do not form a complete 6x6 to 9x9 lower triangle.
    """
    given = _COVARIANCE_NAMES[9] & {field.keyword for field in section.fields}
    for dimension in range(6, 10):
        if given == _COVARIANCE_NAMES[dimension]:
            return dimension
    raise ValueError(
        f"{section.get('OBJECT')} covariance must be a complete 6x6, 7x7, "
        "8x8 or 9x9 lower triangle"
    )


@dataclass(frozen=True)
class CdmObject:
    """Typed view of one object section.

    Parameters
    ----------
    section
        The object section.
    tca
        Time of closest approach (epoch of the state and covariance).
    """

    section: Section
    tca: Time

    @property
    def designator(self) -> str:
        """``OBJECT_DESIGNATOR`` (e.g. the catalogue number)."""
        return _text(self.section, "OBJECT_DESIGNATOR")

    @property
    def name(self) -> str:
        """``OBJECT_NAME``."""
        return _text(self.section, "OBJECT_NAME")

    @property
    def international_designator(self) -> str:
        """``INTERNATIONAL_DESIGNATOR``."""
        return _text(self.section, "INTERNATIONAL_DESIGNATOR")

    @property
    def ref_frame(self) -> str:
        """``REF_FRAME`` of the state vector, as written."""
        return _text(self.section, "REF_FRAME")

    @property
    def hbr(self) -> float | None:
        """Hard-body radius [m] (CDM 2.0 / TraCSS ``HBR``), if given."""
        return _optional_number(self.section, "HBR")

    @property
    def covariance_method(self) -> CovarianceMethod:
        """``COVARIANCE_METHOD``.

        Raises
        ------
        CdmParseError
            If it is neither ``CALCULATED`` nor ``DEFAULT``.
        """
        value = _text(self.section, "COVARIANCE_METHOD")
        try:
            return CovarianceMethod(value)
        except ValueError:
            raise CdmParseError(
                f"COVARIANCE_METHOD must be CALCULATED or DEFAULT, got {value!r}",
                keyword="COVARIANCE_METHOD",
            ) from None

    @property
    def assumed_covariance(self) -> bool:
        """Whether the covariance is assumed (``COVARIANCE_METHOD = DEFAULT``),
        not from orbit determination."""
        return self.covariance_method is CovarianceMethod.DEFAULT

    @property
    def user_defined(self) -> dict[str, str]:
        """``USER_DEFINED_*`` keywords, as text."""
        return {
            keyword: value
            for keyword, value in self.section.extras.items()
            if keyword.startswith(USER_DEFINED_PREFIX)
        }

    def _vector(self, keywords: tuple[str, str, str]) -> FloatArray:
        return np.array([_number(self.section, k) for k in keywords])

    @property
    def position(self) -> FloatArray:
        """Position in ``ref_frame`` [m]."""
        return self._vector(POSITION_KEYWORDS)

    @property
    def velocity(self) -> FloatArray:
        """Velocity in ``ref_frame`` [m/s]."""
        return self._vector(VELOCITY_KEYWORDS)

    @cached_property
    def covariance_rtn(self) -> FloatArray:
        """Full symmetric RTN covariance as given (6x6 to 9x9), unvalidated;
        read-only."""
        matrix = covariance_matrix(
            covariance_dimension(self.section),
            lambda name: _number(self.section, name),
        )
        read_only(matrix)
        return matrix

    def state(self) -> StateVector:
        """State vector at TCA in ``ref_frame``.

        Raises
        ------
        UnsupportedFrame
            If ``REF_FRAME`` is not GCRF, EME2000, ITRF or TEME (a
            :class:`~keepout.errors.GeometryError`).
        """
        return StateVector(
            self.tca, self.position, self.velocity, as_ref_frame(self.ref_frame)
        )

    def covariance(self, *, policy: Policy = "repair") -> Covariance:
        """Validated 6x6 RTN covariance at TCA; repairs are in its ``repairs``.

        Parameters
        ----------
        policy
            ``"repair"`` (default) or ``"raise"``; see
            :func:`keepout.covariance.make_covariance`.
        """
        return make_covariance(
            self.covariance_rtn[:6, :6], "RTN", self.tca, policy=policy
        )

    @property
    def has_covariance(self) -> bool:
        """False if every 6x6 covariance entry is zero.

        Operators send an all-zero covariance when they have none; it means
        "unknown", not "perfectly known".
        """
        return bool(np.any(self.covariance_rtn[:6, :6]))

    def reported(self) -> ReportedObject:
        """The object as reported: state in ``ref_frame``, 6x6 RTN covariance.

        The covariance is ``None`` when the CDM gives an all-zero matrix
        (see :attr:`has_covariance`).

        Raises
        ------
        UnsupportedFrame
            As :meth:`state`.
        """
        return ReportedObject(
            self.designator,
            self.state(),
            self.covariance_rtn[:6, :6] if self.has_covariance else None,
            self.hbr,
        )

    def object_state(self, *, policy: Policy = "repair") -> ObjectState:
        """GCRF object state with covariance; RTN axes from the GCRF state.

        :meth:`Cdm.object_states` converts both objects at once.

        Parameters
        ----------
        policy
            Covariance policy, as in :meth:`covariance`.
        """
        (state,) = gcrf_object_states([self.reported()], policy=policy)
        return state


@dataclass(frozen=True)
class Cdm:
    """A Conjunction Data Message (CCSDS 508.0-B-1, or CDM 2.0).

    Parameters
    ----------
    header
        Header section.
    relative
        Relative metadata/data section (starts at ``TCA``).
    segments
        The ``OBJECT1`` and ``OBJECT2`` sections.

    Raises
    ------
    ValueError
        If a section has the wrong kind, lacks a required keyword, the objects
        are not labelled OBJECT1/OBJECT2 in order, or a covariance is incomplete.
    """

    header: Section
    relative: Section
    segments: tuple[Section, Section]

    def __post_init__(self) -> None:
        if len(self.segments) != 2:
            raise ValueError(f"a CDM has 2 object sections, got {len(self.segments)}")
        named = (
            (self.header, "header", "header"),
            (self.relative, "relative", "relative metadata"),
            (self.segments[0], "segment", "OBJECT1"),
            (self.segments[1], "segment", "OBJECT2"),
        )
        for section, kind, name in named:
            if section.kind != kind:
                raise ValueError(
                    f"{name} section has kind {section.kind!r}, expected {kind!r}"
                )
            missing = section.missing()
            if missing:
                raise ValueError(
                    f"{name} section is missing required keyword(s): "
                    f"{', '.join(missing)}"
                )
        for index, segment in enumerate(self.segments, start=1):
            label = segment.get("OBJECT")
            if label != f"OBJECT{index}":
                raise ValueError(
                    f"segment {index} must have OBJECT = OBJECT{index}, got {label!r}"
                )
            covariance_dimension(segment)

    @property
    def version(self) -> str:
        """``CCSDS_CDM_VERS``, e.g. ``"1.0"`` or ``"2.0"``."""
        return _text(self.header, VERSION_KEYWORD)

    @property
    def creation_date(self) -> Time:
        """``CREATION_DATE``."""
        return _epoch(self.header, "CREATION_DATE")

    @property
    def originator(self) -> str:
        """``ORIGINATOR``."""
        return _text(self.header, "ORIGINATOR")

    @property
    def message_id(self) -> str:
        """``MESSAGE_ID``."""
        return _text(self.header, "MESSAGE_ID")

    @property
    def conjunction_id(self) -> str | None:
        """``CONJUNCTION_ID`` (CDM 2.0), shared by every CDM of one event."""
        return _optional_text(self.header, "CONJUNCTION_ID")

    @property
    def message_for(self) -> str | None:
        """``MESSAGE_FOR``, if given."""
        return _optional_text(self.header, "MESSAGE_FOR")

    @property
    def tca(self) -> Time:
        """Time of closest approach."""
        return _epoch(self.relative, "TCA")

    @property
    def miss_distance(self) -> float:
        """``MISS_DISTANCE`` [m]."""
        return _number(self.relative, "MISS_DISTANCE")

    @property
    def collision_probability(self) -> float | None:
        """Pc reported by the originator, if given."""
        return _optional_number(self.relative, "COLLISION_PROBABILITY")

    @cached_property
    def object1(self) -> CdmObject:
        """Typed view of ``OBJECT1``."""
        return CdmObject(self.segments[0], self.tca)

    @cached_property
    def object2(self) -> CdmObject:
        """Typed view of ``OBJECT2``."""
        return CdmObject(self.segments[1], self.tca)

    @property
    def assumed_covariance(self) -> bool:
        """Whether an object's covariance is assumed
        (:attr:`CdmObject.assumed_covariance`): a Pc from it rests on a model.

        Raises
        ------
        CdmParseError
            If a ``COVARIANCE_METHOD`` is neither ``CALCULATED`` nor
            ``DEFAULT``.
        """
        return self.object1.assumed_covariance or self.object2.assumed_covariance

    def object_states(
        self, *, policy: Policy = "repair"
    ) -> tuple[ObjectState, ObjectState]:
        """Both objects in GCRF (covariance repairs in ``covariance.repairs``).

        Parameters
        ----------
        policy
            Covariance policy; see :func:`keepout.covariance.make_covariance`.
        """
        first, second = gcrf_object_states(
            [self.object1.reported(), self.object2.reported()], policy=policy
        )
        return first, second

    def conjunction(
        self, *, hard_body_radius: float | None = None, policy: Policy = "repair"
    ) -> Conjunction:
        """The conjunction of this message (``OBJECT1`` is the primary).

        Parameters
        ----------
        hard_body_radius
            Combined radius [m]; if ``None``, the sum of both objects' ``HBR``
            (CDM 2.0).
        policy
            Covariance policy; see :func:`keepout.covariance.make_covariance`.

        Raises
        ------
        ValueError
            If no radius is given and an object has no ``HBR``.
        """
        primary, secondary = self.object_states(policy=policy)
        radius = combined_hard_body_radius(primary, secondary, hard_body_radius)
        return Conjunction(primary, secondary, radius)
