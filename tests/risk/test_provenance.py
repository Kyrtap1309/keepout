"""What a Pc rests on: the covariance behind it, or none."""

import astropy.units as u
import pytest
from builders import TYPICAL_HEAD_ON

from keepout.cdm import Cdm, CovarianceMethod, ObjectMetadata, make_cdm
from keepout.risk import PcKind, PcObservation, Provenance
from keepout.risk.provenance import (
    HEADER_COMMENTS,
    covariance_provenance,
    provenance,
)
from keepout.units import HOUR


def cdm(comments: tuple[str, ...] = (), *, assumed: bool = False) -> Cdm:
    method = CovarianceMethod.DEFAULT if assumed else CovarianceMethod.CALCULATED
    return make_cdm(
        TYPICAL_HEAD_ON,
        ObjectMetadata("PRI", "2020-001A", covariance_method=method),
        ObjectMetadata("SEC", "2019-001A"),
        created=TYPICAL_HEAD_ON.tca - 24 * HOUR * u.s,
        comments=comments,
    )


def observation(*, kind: PcKind = "estimate", assumed: bool = False) -> PcObservation:
    return PcObservation(
        24 * HOUR, "keepout (foster)", pc=1e-5, kind=kind, assumed_covariance=assumed
    )


def unread() -> tuple[str, ...]:
    raise AssertionError("the header comments were read")


@pytest.mark.parametrize(
    ("item", "comments", "expected"),
    [
        (observation(assumed=True), HEADER_COMMENTS, Provenance.GP_AGE_MODEL),
        (observation(assumed=True), (), Provenance.ASSUMED),
        (observation(assumed=True), HEADER_COMMENTS[:1], Provenance.ASSUMED),
        (
            observation(assumed=True),
            (*HEADER_COMMENTS, "another comment"),
            Provenance.GP_AGE_MODEL,
        ),
    ],
)
def test_provenance_of_an_assumed_covariance(
    item: PcObservation, comments: tuple[str, ...], expected: Provenance
) -> None:
    assert provenance(item, lambda: comments) is expected


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (observation(kind="upper_bound"), Provenance.MAX_PC),
        (observation(kind="upper_bound", assumed=True), Provenance.MAX_PC),
        (observation(), Provenance.CDM_COVARIANCE),
    ],
)
def test_provenance_reads_no_comments_unless_assumed(
    item: PcObservation, expected: Provenance
) -> None:
    assert provenance(item, unread) is expected


@pytest.mark.parametrize(
    ("assumed", "comments", "expected"),
    [
        (True, HEADER_COMMENTS, Provenance.GP_AGE_MODEL),
        (True, HEADER_COMMENTS[:1], Provenance.ASSUMED),
        (False, HEADER_COMMENTS, Provenance.CDM_COVARIANCE),
    ],
)
def test_covariance_provenance(
    assumed: bool, comments: tuple[str, ...], expected: Provenance
) -> None:
    assert covariance_provenance(cdm(comments, assumed=assumed)) is expected


def test_the_synthetic_operator_marks_its_cdms() -> None:
    from keepout.synthetic import operator

    assert operator.HEADER_COMMENTS is HEADER_COMMENTS
