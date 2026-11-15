import astropy.units as u
import pytest
from builders import TYPICAL_HEAD_ON

from keepout.cdm import Cdm, CovarianceMethod, ObjectMetadata, make_cdm
from keepout.figures import Provenance, covariance_provenance, provenance
from keepout.risk import PcKind, PcObservation
from keepout.synthetic.operator import HEADER_COMMENTS
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


@pytest.mark.parametrize(
    ("item", "comments", "expected"),
    [
        (observation(kind="upper_bound"), (), Provenance.MAX_PC),
        (
            observation(kind="upper_bound", assumed=True),
            HEADER_COMMENTS,
            Provenance.MAX_PC,
        ),
        (observation(assumed=True), HEADER_COMMENTS, Provenance.GP_AGE_MODEL),
        (observation(assumed=True), (), Provenance.ASSUMED),
        (observation(assumed=True), HEADER_COMMENTS[:1], Provenance.ASSUMED),
        (observation(), (), Provenance.CDM_COVARIANCE),
        (observation(), HEADER_COMMENTS, Provenance.CDM_COVARIANCE),
    ],
)
def test_provenance(
    item: PcObservation, comments: tuple[str, ...], expected: Provenance
) -> None:
    assert provenance(item, cdm(comments)) is expected


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


def test_provenance_labels() -> None:
    assert [str(item) for item in Provenance] == [
        "CDM covariance",
        "assumed (GP age model)",
        "assumed covariance",
        "max-Pc (no covariance)",
    ]
