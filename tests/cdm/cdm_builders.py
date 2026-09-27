"""Builders, Hypothesis strategies and comparison helpers for CDM tests."""

import string
from pathlib import Path

import pytest
from astropy.time import Time
from builders import DIAGONAL_COVARIANCE, EPOCH
from hypothesis import strategies as st

from keepout.cdm.keywords import (
    REQUIRED,
    SEGMENT,
    covariance_entries,
    covariance_names,
    parse_epoch,
)
from keepout.cdm.model import Cdm, Field, Section

DATA = Path(__file__).parent / "data"
B1_KVN = DATA / "b1_example.kvn"
B1_XML = DATA / "b1_example.xml"
TRACSS = DATA / "tracss_v2.kvn"
TCA = EPOCH
CREATED = parse_epoch("2010-03-12T22:31:12.000")
POSITIONS = {1: (7000e3, 0.0, 0.0), 2: (7000e3, 0.0, 500.0)}
VELOCITIES = {1: (0.0, 7500.0, 0.0), 2: (0.0, 0.0, 7500.0)}


def segment_fields(index: int, frame: str = "EME2000") -> list[Field]:
    """Minimal valid fields of object section ``index`` (1 or 2)."""
    fields = [
        Field("OBJECT", f"OBJECT{index}"),
        Field("OBJECT_DESIGNATOR", f"1000{index}"),
        Field("CATALOG_NAME", "SATCAT"),
        Field("OBJECT_NAME", f"SAT {index}"),
        Field("INTERNATIONAL_DESIGNATOR", f"2000-00{index}A"),
        Field("EPHEMERIS_NAME", "NONE"),
        Field("COVARIANCE_METHOD", "CALCULATED"),
        Field("MANEUVERABLE", "NO"),
        Field("REF_FRAME", frame),
    ]
    values = (*POSITIONS[index], *VELOCITIES[index])
    names = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
    fields += [Field(name, value) for name, value in zip(names, values, strict=True)]
    fields += [
        Field(name, float(DIAGONAL_COVARIANCE[row, column]))
        for name, row, column in covariance_entries(6)
    ]
    return fields


def minimal_cdm(
    segment1: list[Field] | None = None, segment2: list[Field] | None = None
) -> Cdm:
    """A minimal valid CDM; object sections can be replaced."""
    header = Section(
        "header",
        (
            Field("CCSDS_CDM_VERS", "1.0"),
            Field("CREATION_DATE", CREATED),
            Field("ORIGINATOR", "KEEPOUT"),
            Field("MESSAGE_ID", "TEST-1"),
        ),
    )
    relative = Section("relative", (Field("TCA", TCA), Field("MISS_DISTANCE", 500.0)))
    first = segment_fields(1) if segment1 is None else segment1
    second = segment_fields(2) if segment2 is None else segment2
    return Cdm(
        header,
        relative,
        (Section("segment", tuple(first)), Section("segment", tuple(second))),
    )


def replace(fields: list[Field], keyword: str, value: str | float) -> list[Field]:
    """Copy of ``fields`` with the value of ``keyword`` replaced."""
    return [
        Field(field.keyword, value, field.unit) if field.keyword == keyword else field
        for field in fields
    ]


def without(fields: list[Field], keyword: str) -> list[Field]:
    """Copy of ``fields`` without ``keyword``."""
    return [field for field in fields if field.keyword != keyword]


def assert_equivalent(actual: Cdm, expected: Cdm) -> None:
    """Same sections, keywords, order and units; values equal to a few ulps."""
    actual_sections = (actual.header, actual.relative, *actual.segments)
    expected_sections = (expected.header, expected.relative, *expected.segments)
    for got, want in zip(actual_sections, expected_sections, strict=True):
        assert got.kind == want.kind
        assert [f.keyword for f in got.fields] == [f.keyword for f in want.fields]
        for got_field, want_field in zip(got.fields, want.fields, strict=True):
            name = want_field.keyword
            assert got_field.unit == want_field.unit, name
            if isinstance(want_field.value, Time):
                assert isinstance(got_field.value, Time), name
                offset = (got_field.value - want_field.value).to_value("s")
                assert abs(float(offset)) < 1e-9, name
            elif isinstance(want_field.value, float):
                assert got_field.value == pytest.approx(
                    want_field.value, rel=1e-14, abs=0.0
                ), name
            else:
                assert got_field.value == want_field.value, name


words = (
    st.text(alphabet=string.ascii_uppercase + string.digits + " -._", max_size=16)
    .map(str.strip)
    .filter(bool)
)
reals = st.floats(
    min_value=-1e12, max_value=1e12, allow_nan=False, allow_subnormal=False
).filter(lambda x: x == 0.0 or abs(x) > 1e-200)


@st.composite
def epochs(draw: st.DrawFn) -> Time:
    """UTC epochs 1990-2025 with 0 to 6 fractional digits."""
    date = (
        f"{draw(st.integers(1990, 2025)):04d}-{draw(st.integers(1, 12)):02d}-"
        f"{draw(st.integers(1, 28)):02d}"
    )
    clock = (
        f"{draw(st.integers(0, 23)):02d}:{draw(st.integers(0, 59)):02d}:"
        f"{draw(st.integers(0, 59)):02d}"
    )
    digits = draw(st.text(alphabet=string.digits, max_size=6))
    return parse_epoch(f"{date}T{clock}" + (f".{digits}" if digits else ""))


def comments() -> st.SearchStrategy[list[Field]]:
    return st.lists(words.map(lambda text: Field("COMMENT", text)), max_size=2)


def extras(prefix: str) -> st.SearchStrategy[list[Field]]:
    units = st.none() | st.sampled_from(["m", "kg", "n/a"])
    return st.lists(st.tuples(words, units), max_size=2).map(
        lambda items: [
            Field(f"{prefix}_{index}", value, unit)
            for index, (value, unit) in enumerate(items)
        ]
    )


_REQUIRED_TEXT = tuple(
    name
    for name in REQUIRED["segment"]
    if SEGMENT[name].kind == "str" and name not in ("OBJECT", "REF_FRAME")
)


@st.composite
def segments(draw: st.DrawFn, index: int) -> Section:
    # Comments open the section, where both KVN and NDM/XML put them.
    fields = [*draw(comments()), Field("OBJECT", f"OBJECT{index}")]
    for name in _REQUIRED_TEXT:
        fields.append(Field(name, draw(words)))
    fields.append(
        Field("REF_FRAME", draw(st.sampled_from(["EME2000", "GCRF", "ITRF"])))
    )
    if draw(st.booleans()):
        fields.append(Field("TIME_LASTOB_START", draw(epochs())))
    if draw(st.booleans()):
        fields.append(Field("OBS_USED", draw(st.integers(0, 10**6))))
    if draw(st.booleans()):
        fields.append(Field("HBR", draw(reals)))
    for name in ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"):
        fields.append(Field(name, draw(reals)))
    for name in covariance_names(draw(st.integers(6, 9))):
        fields.append(Field(name, draw(reals)))
    fields += draw(extras("USER_DEFINED_TEST"))
    fields += draw(extras("X_KEEPOUT_SEG"))
    return Section("segment", tuple(fields))


@st.composite
def cdms(draw: st.DrawFn) -> Cdm:
    """Valid CDMs (any covariance values) exercising comments and extras."""
    header = Section(
        "header",
        (
            Field("CCSDS_CDM_VERS", draw(st.sampled_from(["1.0", "2.0"]))),
            *draw(comments()),
            Field("CREATION_DATE", draw(epochs())),
            Field("ORIGINATOR", draw(words)),
            Field("MESSAGE_ID", draw(words)),
            *draw(extras("X_KEEPOUT_HDR")),
        ),
    )
    relative_fields = [
        *draw(comments()),
        Field("TCA", draw(epochs())),
        Field("MISS_DISTANCE", draw(reals)),
    ]
    if draw(st.booleans()):
        for axis in "RTN":
            relative_fields.append(Field(f"RELATIVE_POSITION_{axis}", draw(reals)))
    if draw(st.booleans()):
        relative_fields.append(Field("COLLISION_PROBABILITY", draw(reals)))
    relative_fields += draw(extras("X_KEEPOUT_REL"))
    return Cdm(
        header,
        Section("relative", tuple(relative_fields)),
        (draw(segments(1)), draw(segments(2))),
    )
