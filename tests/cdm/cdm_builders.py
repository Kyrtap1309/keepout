"""Builders and comparison helpers for CDM tests (the Hypothesis strategies
are in ``cdm_strategies``)."""

import pytest
from astropy.time import Time
from builders import CDM_DATA, DIAGONAL_COVARIANCE, EPOCH

from keepout.ccsds.epochs import parse_epoch
from keepout.cdm.keywords import covariance_entries
from keepout.cdm.model import Cdm, Field, Section

B1_XML = CDM_DATA / "b1_example.xml"
TRACSS = CDM_DATA / "tracss_v2.kvn"
TCA = EPOCH
CREATED = parse_epoch("2010-03-12T22:31:12.000")


def mutate(text: str, old: str, new: str, count: int = 1) -> str:
    """``text`` with ``old`` replaced, asserting it occurs exactly ``count`` times."""
    assert text.count(old) == count, old
    return text.replace(old, new)


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
    actual_sections = (actual.header, actual.relative, *actual.object_sections)
    expected_sections = (expected.header, expected.relative, *expected.object_sections)
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
