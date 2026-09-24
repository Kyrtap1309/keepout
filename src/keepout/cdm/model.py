"""In-memory Conjunction Data Message.

A :class:`Cdm` keeps every field in message order — comments and unknown
keywords included — so KVN and XML round-trips are lossless. Known keywords
hold typed SI values (see :mod:`keepout.cdm.keywords`); unknown keywords hold
their raw text and unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.time import Time

from keepout._types import REF_FRAMES, FloatArray, StateVector
from keepout.cdm.keywords import (
    COVARIANCE_AXES,
    KEYWORDS,
    REQUIRED,
    FieldValue,
    SectionKind,
    ValueKind,
    covariance_names,
)
from keepout.covariance import Covariance, Policy, make_covariance
from keepout.errors import GeometryError
from keepout.frames import to_gcrf
from keepout.objects import ObjectState

COMMENT = "COMMENT"
USER_DEFINED_PREFIX = "USER_DEFINED_"


@dataclass(frozen=True)
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


def _matches(value: object, kind: ValueKind) -> bool:
    if kind == "time":
        return isinstance(value, Time) and bool(value.isscalar)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, float)
    return isinstance(value, str)


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
    ValueError
        For an unknown kind, a repeated keyword, a value of the wrong type or
        a unit on a known keyword.
    """

    kind: SectionKind
    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        if self.kind not in KEYWORDS:
            raise ValueError(f"unknown section kind {self.kind!r}")
        table = KEYWORDS[self.kind]
        seen: set[str] = set()
        for field in self.fields:
            if field.keyword != COMMENT:
                if field.keyword in seen:
                    raise ValueError(
                        f"{self.kind} section repeats keyword {field.keyword}"
                    )
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

    def get(self, keyword: str) -> FieldValue | None:
        """Value of the first field with this keyword, or None.

        Parameters
        ----------
        keyword
            Keyword to look up.
        """
        for field in self.fields:
            if field.keyword == keyword:
                return field.value
        return None

    @property
    def comments(self) -> tuple[str, ...]:
        """Comment texts in order."""
        return tuple(str(f.value) for f in self.fields if f.keyword == COMMENT)

    @property
    def extras(self) -> dict[str, str]:
        """Keywords not in the CCSDS 508.0-B-1 table (plus HBR), as text."""
        table = KEYWORDS[self.kind]
        return {
            f.keyword: str(f.value)
            for f in self.fields
            if f.keyword != COMMENT and f.keyword not in table
        }

    def missing(self) -> tuple[str, ...]:
        """Required keywords absent from this section."""
        return tuple(k for k in REQUIRED[self.kind] if self.get(k) is None)


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
    present = {field.keyword for field in section.fields}
    dimension = 0
    for size in range(6, 10):
        if set(covariance_names(size)) <= present:
            dimension = size
    given = set(covariance_names(9)) & present
    if dimension == 0 or given != set(covariance_names(dimension)):
        raise ValueError(
            f"{section.get('OBJECT')} covariance must be a complete 6x6, 7x7, "
            "8x8 or 9x9 lower triangle"
        )
    return dimension


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
    def user_defined(self) -> dict[str, str]:
        """``USER_DEFINED_*`` keywords, as text."""
        return {
            keyword: value
            for keyword, value in self.section.extras.items()
            if keyword.startswith(USER_DEFINED_PREFIX)
        }

    @property
    def position(self) -> FloatArray:
        """Position in ``ref_frame`` [m]."""
        return np.array(
            [_number(self.section, k) for k in ("X", "Y", "Z")], dtype=np.float64
        )

    @property
    def velocity(self) -> FloatArray:
        """Velocity in ``ref_frame`` [m/s]."""
        return np.array(
            [_number(self.section, k) for k in ("X_DOT", "Y_DOT", "Z_DOT")],
            dtype=np.float64,
        )

    @property
    def covariance_rtn(self) -> FloatArray:
        """Full symmetric RTN covariance as given (6x6 to 9x9), unvalidated."""
        dimension = covariance_dimension(self.section)
        matrix = np.zeros((dimension, dimension))
        for row in range(dimension):
            for column in range(row + 1):
                name = f"C{COVARIANCE_AXES[row]}_{COVARIANCE_AXES[column]}"
                matrix[row, column] = matrix[column, row] = _number(self.section, name)
        return matrix

    def state(self) -> StateVector:
        """State vector at TCA in ``ref_frame``.

        Raises
        ------
        GeometryError
            If ``REF_FRAME`` is not GCRF, EME2000 or ITRF.
        """
        frame = self.ref_frame
        if frame not in REF_FRAMES:
            raise GeometryError(
                f"unsupported REF_FRAME {frame!r}; keepout supports "
                f"{', '.join(REF_FRAMES)}"
            )
        return StateVector(self.tca, self.position, self.velocity, frame)

    def covariance(
        self, *, policy: Policy = "repair"
    ) -> tuple[Covariance, tuple[str, ...]]:
        """Validated 6x6 RTN covariance at TCA.

        Parameters
        ----------
        policy
            ``"repair"`` (default) or ``"raise"``; see
            :func:`keepout.covariance.make_covariance`.
        """
        return make_covariance(
            self.covariance_rtn[:6, :6], "RTN", self.tca, policy=policy
        )

    def object_state(
        self, *, policy: Policy = "repair"
    ) -> tuple[ObjectState, tuple[str, ...]]:
        """GCRF object state with covariance; RTN axes from the GCRF state.

        Parameters
        ----------
        policy
            Covariance policy, as in :meth:`covariance`.
        """
        state = to_gcrf(self.state())
        covariance, diagnostics = self.covariance(policy=policy)
        return (
            ObjectState(self.designator, state, covariance.in_gcrf(state), self.hbr),
            diagnostics,
        )


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
        return _text(self.header, "CCSDS_CDM_VERS")

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

    @property
    def object1(self) -> CdmObject:
        """Typed view of ``OBJECT1``."""
        return CdmObject(self.segments[0], self.tca)

    @property
    def object2(self) -> CdmObject:
        """Typed view of ``OBJECT2``."""
        return CdmObject(self.segments[1], self.tca)

    def object_states(
        self, *, policy: Policy = "repair"
    ) -> tuple[ObjectState, ObjectState, tuple[str, ...]]:
        """Both objects in GCRF, with diagnostics labelled by object.

        Parameters
        ----------
        policy
            Covariance policy; see :func:`keepout.covariance.make_covariance`.
        """
        first, first_diagnostics = self.object1.object_state(policy=policy)
        second, second_diagnostics = self.object2.object_state(policy=policy)
        diagnostics = tuple(f"OBJECT1: {d}" for d in first_diagnostics) + tuple(
            f"OBJECT2: {d}" for d in second_diagnostics
        )
        return first, second, diagnostics
