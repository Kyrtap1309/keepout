"""Building a CDM from a conjunction: fields, determinism and round-trips."""

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypedDict, Unpack
from xml.etree import ElementTree

import astropy.units as u
import numpy as np
import pytest
import xmlschema
from astropy.time import Time
from builders import DIAGONAL_COVARIANCE, LEO_R, LEO_V, known
from cdm_builders import assert_equivalent, epochs, words
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from screening_data import load_fixture

from keepout._types import FloatArray
from keepout.cdm import Cdm, Section, kvn, xml
from keepout.cdm.builder import (
    ORIGINATOR,
    SCREEN_VOLUME_COMMENT,
    CovarianceMethod,
    Maneuverable,
    ObjectMetadata,
    ObjectType,
    RtnSemiAxes,
    make_cdm,
    make_message_id,
)
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.events import ConjunctionEvent
from keepout.frames import relative_position_rtn
from keepout.objects import ObjectState
from keepout.screening import ScreeningVolume
from keepout.state import StateVector
from keepout.synthetic import SyntheticCovarianceModel

SCHEMA = xmlschema.XMLSchema(Path(__file__).parent / "xsd" / "cdm.xsd")
"""NDM/XML 2.0.0 CDM module (CCSDS 508.0-B-1): version 1.0 messages only."""

TCA = Time("2026-10-08T12:34:56.789012", scale="utc", precision=6)
CREATED = Time("2026-10-06T00:00:00.000", scale="utc", precision=3)
SECONDARY_R = LEO_R + np.array([120.0, -450.0, 80.0])
SECONDARY_V = np.array([0.0, -7546.0 * 0.5, 7546.0 * 0.8])

ISS = ObjectMetadata(
    "ISS (ZARYA)",
    "1998-067A",
    object_type=ObjectType.PAYLOAD,
    maneuverable=Maneuverable.YES,
)
DEBRIS = ObjectMetadata("FENGYUN 1C DEB", "1999-025AQZ")


def gcrf_object(
    object_id: str,
    r: FloatArray,
    v: FloatArray,
    covariance: FloatArray | None,
    radius: float | None,
    epoch: Time = TCA,
) -> ObjectState:
    state = StateVector(epoch, r, v, "GCRF")
    gcrf = None if covariance is None else Covariance(covariance, "GCRF", epoch)
    return ObjectState(object_id, state, gcrf, radius)


def conjunction(
    first: float | None = 5.0,
    second: float | None = 2.5,
    combined: float = 7.5,
    secondary_covariance: FloatArray | None = DIAGONAL_COVARIANCE,
) -> Conjunction:
    covariance = np.array(DIAGONAL_COVARIANCE)
    covariance[0, 1] = covariance[1, 0] = 50.0
    return Conjunction(
        gcrf_object("25544", LEO_R, LEO_V, covariance, first),
        gcrf_object("49863", SECONDARY_R, SECONDARY_V, secondary_covariance, second),
        combined,
    )


class Options(TypedDict, total=False):
    """The keyword arguments of :func:`make_cdm` besides ``created``."""

    originator: str
    message_id: str | None
    message_for: str | None
    screening_period: tuple[Time, Time] | None
    screening_volume: RtnSemiAxes | None
    comments: Sequence[str]


def example(created: Time = CREATED, **options: Unpack[Options]) -> Cdm:
    return make_cdm(conjunction(), ISS, DEBRIS, created=created, **options)


def ccsds_text(text: str) -> Any:
    """``text`` where an enumeration is typed: :class:`ObjectMetadata` also
    takes the CCSDS text of its enumerations."""
    return text


def assert_same_conjunction(actual: Conjunction, expected: Conjunction) -> None:
    """Same ids, radii, states (at the TCA) and covariances."""
    assert actual.combined_hard_body_radius == pytest.approx(
        expected.combined_hard_body_radius, rel=1e-12
    )
    for got, want in zip(
        (actual.primary, actual.secondary),
        (expected.primary, expected.secondary),
        strict=True,
    ):
        assert got.object_id == want.object_id
        assert got.hard_body_radius == want.hard_body_radius
        assert got.state.frame == "GCRF"
        offset = float((got.state.epoch - want.state.epoch).to_value("s"))
        assert abs(offset) <= 0.5e-6
        # Written in km: a value in m may come back 1 ulp off.
        np.testing.assert_allclose(got.state.r, want.state.r, rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(got.state.v, want.state.v, rtol=1e-15, atol=0.0)
        if want.covariance is None:
            assert got.covariance is None
            continue
        expected_matrix = want.covariance.matrix
        scale = float(np.max(np.abs(expected_matrix)))
        np.testing.assert_allclose(
            known(got.covariance).matrix,
            expected_matrix,
            rtol=1e-9,
            atol=1e-12 * scale,
        )
        assert known(got.covariance).repairs == ()
        assert known(got.covariance).position_only == want.covariance.position_only


def test_the_message_reads_back_to_the_conjunction() -> None:
    cdm = example()
    assert_same_conjunction(cdm.conjunction(), conjunction())
    assert_same_conjunction(kvn.loads(kvn.dumps(cdm)).conjunction(), conjunction())
    assert_same_conjunction(xml.loads(xml.dumps(cdm)).conjunction(), conjunction())


def test_the_header_is_keepouts_version_2_message_by_default() -> None:
    cdm = example()
    assert cdm.version == "2.0"
    assert cdm.originator == ORIGINATOR == "KEEPOUT"
    assert cdm.creation_date is CREATED
    assert cdm.message_id == "25544_49863_20261008T123456.789Z_20261006T000000.000Z"
    assert cdm.message_for is None
    assert cdm.header.comments == ()
    assert [f.keyword for f in cdm.header.fields] == [
        "CCSDS_CDM_VERS",
        "CREATION_DATE",
        "ORIGINATOR",
        "MESSAGE_ID",
    ]


def test_the_header_takes_the_given_originator_id_recipient_and_comments() -> None:
    cdm = example(
        originator="OPS",
        message_id="M-1",
        message_for="ACME SAT OPS",
        comments=["Synthetic operator run", "Not for operations"],
    )
    assert (cdm.originator, cdm.message_id, cdm.message_for) == (
        "OPS",
        "M-1",
        "ACME SAT OPS",
    )
    assert cdm.header.comments == ("Synthetic operator run", "Not for operations")


def test_each_object_section_carries_its_metadata() -> None:
    cdm = example()
    first, second = cdm.object1, cdm.object2
    assert (first.object_id, first.name, first.international_designator) == (
        "25544",
        "ISS (ZARYA)",
        "1998-067A",
    )
    assert first.section.get("OBJECT_TYPE") == "PAYLOAD"
    assert first.section.get("MANEUVERABLE") == "YES"
    assert second.section.get("OBJECT_TYPE") is None
    assert second.section.get("MANEUVERABLE") == "N/A"
    for item in (first, second):
        assert item.ref_frame == "GCRF"
        assert item.section.get("CATALOG_NAME") == "SATCAT"
        assert item.section.get("EPHEMERIS_NAME") == "NONE"
        assert item.section.get("COVARIANCE_METHOD") == "CALCULATED"


def test_each_object_section_carries_its_state_radius_and_rtn_covariance() -> None:
    cdm = example()
    first, second = cdm.object1, cdm.object2
    assert (first.hard_body_radius, second.hard_body_radius) == (5.0, 2.5)
    np.testing.assert_array_equal(first.position, LEO_R)
    np.testing.assert_array_equal(second.velocity, SECONDARY_V)
    # LEO_R lies along x and LEO_V along y: the primary's RTN axes are GCRF's.
    np.testing.assert_allclose(
        first.covariance_rtn, known(conjunction().primary.covariance).matrix
    )


def numbers(section: Section, *keywords: str) -> list[float]:
    """The float values of ``keywords`` in ``section``."""
    found: list[float] = []
    for keyword in keywords:
        value = section.get(keyword)
        assert isinstance(value, float), keyword
        found.append(value)
    return found


def test_relative_fields_are_in_the_primary_rtn_frame() -> None:
    cdm = example()
    relative = cdm.relative
    assert cdm.tca.precision == 6
    assert abs(float((cdm.tca - TCA).to_value("s"))) < 1e-9
    miss = relative_position_rtn(LEO_R, LEO_V, SECONDARY_R)
    position = numbers(relative, *(f"RELATIVE_POSITION_{axis}" for axis in "RTN"))
    np.testing.assert_allclose(position, miss, rtol=1e-12)
    assert cdm.miss_distance == pytest.approx(float(np.linalg.norm(miss)), rel=1e-12)
    # LEO_R lies along x and LEO_V along y: R, T, N are x, y, z.
    velocity = numbers(relative, *(f"RELATIVE_VELOCITY_{axis}" for axis in "RTN"))
    np.testing.assert_allclose(velocity, SECONDARY_V - LEO_V, rtol=1e-12)
    speed = relative.get("RELATIVE_SPEED")
    assert speed == pytest.approx(float(np.linalg.norm(SECONDARY_V - LEO_V)), rel=1e-12)
    assert relative.comments == ()
    assert relative.get("START_SCREEN_PERIOD") is None
    assert relative.get("SCREEN_VOLUME_X") is None


def test_the_screening_period_and_volume_are_written_with_their_comment() -> None:
    start = Time("2026-10-06T00:00:00", scale="utc")
    stop = start + 7 * u.day
    volume = ScreeningVolume(1000.0, 5000.0, 2000.0)
    relative = example(screening_period=(start, stop), screening_volume=volume).relative
    assert relative.comments == (SCREEN_VOLUME_COMMENT,)
    assert relative.get("START_SCREEN_PERIOD") is start
    assert relative.get("STOP_SCREEN_PERIOD") is stop
    assert [
        relative.get(name)
        for name in (
            "SCREEN_VOLUME_FRAME",
            "SCREEN_VOLUME_SHAPE",
            "SCREEN_VOLUME_X",
            "SCREEN_VOLUME_Y",
            "SCREEN_VOLUME_Z",
        )
    ] == ["RTN", "ELLIPSOID", 1000.0, 5000.0, 2000.0]


def test_the_screening_period_must_end_after_it_starts() -> None:
    with pytest.raises(ValueError, match="must end after it starts"):
        example(screening_period=(TCA, TCA))


def test_the_same_inputs_give_the_same_message() -> None:
    assert example().message_id == example().message_id
    assert kvn.dumps(example()) == kvn.dumps(example())


def test_message_id_times_are_utc_to_the_millisecond() -> None:
    # The times are written in UTC whatever their scale (TT - UTC = 69.184 s).
    tt = Time("2026-10-08T12:36:05.973", scale="tt")
    created = Time("2026-10-06T00:01:09.1844", scale="tt")
    assert make_message_id("1", "2", tt, created) == (
        "1_2_20261008T123456.789Z_20261006T000000.000Z"
    )


def test_message_id_rounds_up_into_the_next_second() -> None:
    tca = Time("2026-10-08T12:34:59.9996", scale="utc")
    assert make_message_id("25544", "49863", tca, CREATED) == (
        "25544_49863_20261008T123500.000Z_20261006T000000.000Z"
    )


def test_messages_issued_at_different_times_are_successive_cdms_of_an_event() -> None:
    first = example()
    later = example(created=CREATED + 1 * u.hour)
    assert later.message_id == "25544_49863_20261008T123456.789Z_20261006T010000.000Z"
    event = ConjunctionEvent.from_cdms([later, first])
    assert event.cdms == (first, later)


def test_the_tca_of_the_conjunction_is_not_changed() -> None:
    item = conjunction()
    precision = item.tca.precision
    make_cdm(item, ISS, DEBRIS, created=CREATED)
    assert item.tca.precision == precision


def without_radii() -> tuple[Conjunction, Cdm]:
    """A conjunction without per-object radii and its message, with comments
    in the header, the object sections and a covariance block."""
    item = conjunction(first=None, second=None)
    cdm = make_cdm(
        item,
        dataclasses.replace(ISS, comments=("Fleet object",)),
        dataclasses.replace(
            DEBRIS,
            object_type=ObjectType.DEBRIS,
            covariance_method=CovarianceMethod.DEFAULT,
            covariance_comments=("Assumed covariance", "Second line"),
        ),
        created=CREATED,
        message_for="ACME",
        comments=["Header note"],
        screening_period=(CREATED, CREATED + 7 * u.day),
        screening_volume=ScreeningVolume.sphere(5000.0),
    )
    return item, cdm


def test_without_radii_the_message_is_version_1() -> None:
    _, cdm = without_radii()
    assert cdm.version == "1.0"
    assert (cdm.object1.hard_body_radius, cdm.object2.hard_body_radius) == (None, None)


def test_a_version_1_message_is_valid_ccsds_xml() -> None:
    _, cdm = without_radii()
    SCHEMA.validate(xml.dumps(cdm))


def test_object_and_covariance_comments_open_their_blocks() -> None:
    _, cdm = without_radii()
    root = ElementTree.fromstring(xml.dumps(cdm))
    matrix = root.find("body/segment[2]/data/covarianceMatrix")
    assert matrix is not None
    assert [e.tag for e in matrix[:3]] == ["COMMENT", "COMMENT", "CR_R"]
    assert [e.text for e in matrix[:2]] == ["Assumed covariance", "Second line"]
    metadata = root.find("body/segment[1]/metadata")
    assert metadata is not None
    assert (metadata[0].tag, metadata[0].text) == ("COMMENT", "Fleet object")


def test_without_radii_the_conjunction_needs_the_combined_radius() -> None:
    item, cdm = without_radii()
    with pytest.raises(ValueError, match="no hard-body radius for 25544, 49863"):
        cdm.conjunction()
    assert_same_conjunction(cdm.conjunction(combined_hard_body_radius=7.5), item)


def test_a_version_1_message_round_trips() -> None:
    item, cdm = without_radii()
    read = xml.loads(xml.dumps(cdm))
    assert_equivalent(read, cdm)
    assert_equivalent(kvn.loads(kvn.dumps(cdm)), cdm)
    assert_same_conjunction(read.conjunction(combined_hard_body_radius=7.5), item)


def test_an_unknown_covariance_is_written_as_zeros() -> None:
    item = conjunction(secondary_covariance=None)
    cdm = make_cdm(item, ISS, DEBRIS, created=CREATED)
    assert not cdm.object2.has_covariance
    np.testing.assert_array_equal(cdm.object2.covariance_rtn, np.zeros((6, 6)))
    assert cdm.conjunction().secondary.covariance is None


def test_a_covariance_known_only_in_position_stays_so() -> None:
    position = Covariance.from_position(np.diag([4e4, 1.6e7, 9e4]), "RTN", TCA)
    base = conjunction()
    secondary = dataclasses.replace(
        base.secondary, covariance=position.in_gcrf(base.secondary.state)
    )
    item = Conjunction(base.primary, secondary, base.combined_hard_body_radius)
    cdm = make_cdm(item, ISS, DEBRIS, created=CREATED)
    np.testing.assert_allclose(
        cdm.object2.covariance_rtn[:3, :3], position.position, rtol=1e-12, atol=1e-6
    )
    assert not np.any(cdm.object2.covariance_rtn[3:])
    assert cdm.conjunction().position_only_covariance == ("secondary",)


@pytest.mark.parametrize(
    ("first", "second", "combined"),
    [(5.0, None, 7.5), (None, 2.5, 7.5), (5.0, 2.5, 20.0)],
)
def test_the_radii_must_be_per_object(
    first: float | None, second: float | None, combined: float
) -> None:
    with pytest.raises(ValueError, match="must add up to the combined radius"):
        make_cdm(conjunction(first, second, combined), ISS, DEBRIS, created=CREATED)


def test_metadata_takes_the_ccsds_text_of_its_enumerations() -> None:
    metadata = ObjectMetadata(
        "SAT",
        "UNKNOWN",
        object_type=ccsds_text("ROCKET BODY"),
        maneuverable=ccsds_text("NO"),
        covariance_method=ccsds_text("DEFAULT"),
    )
    assert metadata.object_type is ObjectType.ROCKET_BODY
    assert metadata.maneuverable is Maneuverable.NO
    assert metadata.covariance_method is CovarianceMethod.DEFAULT


def test_an_unknown_manoeuvrability_is_written_n_a() -> None:
    assert str(Maneuverable.NOT_APPLICABLE) == "N/A"


@pytest.mark.parametrize(
    ("keyword", "text"),
    [
        ("maneuverable", "MAYBE"),
        ("covariance_method", "ESTIMATED"),
        ("object_type", "UFO"),
    ],
)
def test_metadata_rejects_values_ccsds_does_not_define(keyword: str, text: str) -> None:
    with pytest.raises(ValueError, match="is not a valid"):
        ObjectMetadata("SAT", "2000-001A", **{keyword: ccsds_text(text)})


COMMENTED: dict[str, tuple[Callable[[str], object], str]] = {
    "object": (
        lambda text: ObjectMetadata("SAT", "2000-001A", comments=(text,)),
        r"object comment .* must be one line",
    ),
    "covariance": (
        lambda text: ObjectMetadata("SAT", "2000-001A", covariance_comments=(text,)),
        "covariance comment",
    ),
    "header": (lambda text: example(comments=[text]), "header comment"),
}
"""Where a comment goes: what builds it there and the error it raises."""


@pytest.mark.parametrize("where", list(COMMENTED))
@pytest.mark.parametrize("text", ["two\nlines", " leading", "trailing ", "a\rb"])
def test_comments_must_survive_a_round_trip(where: str, text: str) -> None:
    build, message = COMMENTED[where]
    with pytest.raises(ValueError, match=message):
        build(text)


def test_an_empty_required_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing required keyword"):
        make_cdm(
            conjunction(), ObjectMetadata("", "1998-067A"), DEBRIS, created=CREATED
        )


def test_the_synthetic_model_comments_fit_a_covariance_block() -> None:
    (leo,) = (omm for omm in load_fixture() if omm.catalog_number == 900)
    model = SyntheticCovarianceModel()
    metadata = dataclasses.replace(
        ISS,
        covariance_method=CovarianceMethod.DEFAULT,
        covariance_comments=model.comments(leo, TCA),
    )
    cdm = make_cdm(conjunction(), metadata, DEBRIS, created=CREATED)
    text = kvn.dumps(cdm)
    assert "COMMENT Synthetic covariance: keepout GP element-set age model" in text
    assert kvn.loads(text).object1.section.comments == model.comments(leo, TCA)


# Property: any conjunction round-trips through the message.


def floats(bound: float) -> st.SearchStrategy[float]:
    """Floats in ``[-bound, bound]``, those closer to 0 than a millionth set to
    0 (the message writes metres as km: a subnormal km value would not
    round-trip)."""
    return st.floats(-bound, bound).map(lambda x: x if abs(x) > 1e-6 else 0.0)


coordinates = floats(5e7)
offsets = floats(1e5)
speeds = floats(1e4)
SCALES = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0])


@st.composite
def covariances(draw: st.DrawFn) -> FloatArray | None:
    """Unknown, position-only or full covariances, well conditioned."""
    kind = draw(st.sampled_from(["unknown", "position", "full"]))
    if kind == "unknown":
        return None
    root = draw(arrays(np.float64, (6, 6), elements=st.floats(-1.0, 1.0)))
    matrix: FloatArray = (root @ root.T + np.eye(6)) * np.outer(SCALES, SCALES)
    if kind == "position":
        matrix[3:] = matrix[:, 3:] = 0.0
    return matrix


@st.composite
def conjunctions(draw: st.DrawFn) -> Conjunction:
    epoch = draw(epochs())
    r = draw(arrays(np.float64, 3, elements=coordinates))
    v = draw(arrays(np.float64, 3, elements=speeds))
    r2 = r + draw(arrays(np.float64, 3, elements=offsets))
    v2 = v + draw(arrays(np.float64, 3, elements=speeds))
    # Well-defined RTN frames: positions not small, velocities not along them.
    for position, velocity in ((r, v), (r2, v2)):
        size = float(np.linalg.norm(position))
        assume(size > 6e6)
        normal = float(np.linalg.norm(np.cross(position, velocity)))
        assume(normal > 1e-3 * size * float(np.linalg.norm(velocity)))
    radius = st.floats(0.05, 50.0)
    radii = draw(st.none() | st.tuples(radius, radius))
    first, second = (None, None) if radii is None else radii
    combined = draw(st.floats(0.1, 100.0)) if radii is None else sum(radii)
    ids = draw(st.lists(words, min_size=2, max_size=2, unique=True))
    return Conjunction(
        gcrf_object(ids[0], r, v, draw(covariances()), first, epoch),
        gcrf_object(ids[1], r2, v2, draw(covariances()), second, epoch),
        combined,
    )


metadata = st.builds(
    ObjectMetadata,
    words,
    words,
    catalog_name=words,
    object_type=st.none() | st.sampled_from(ObjectType),
    maneuverable=st.sampled_from(Maneuverable),
    covariance_method=st.sampled_from(CovarianceMethod),
    ephemeris_name=words,
    comments=st.lists(words, max_size=2).map(tuple),
    covariance_comments=st.lists(words, max_size=2).map(tuple),
)


@settings(max_examples=50, deadline=None)
@given(conjunctions(), metadata, metadata)
def test_any_conjunction_round_trips_through_the_message(
    item: Conjunction, primary: ObjectMetadata, secondary: ObjectMetadata
) -> None:
    cdm = make_cdm(item, primary, secondary, created=CREATED)
    radius = (
        None
        if cdm.object1.hard_body_radius is not None
        else item.combined_hard_body_radius
    )
    for message in (cdm, kvn.loads(kvn.dumps(cdm)), xml.loads(xml.dumps(cdm))):
        assert_equivalent(message, cdm)
        assert_same_conjunction(
            message.conjunction(combined_hard_body_radius=radius), item
        )
