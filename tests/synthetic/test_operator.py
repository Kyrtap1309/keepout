"""The synthetic operator: a fleet screened against the fixture, a CDM per
approach with the model's covariances, and the risk of each, judged from
the CDM."""

import dataclasses
import functools
from collections import Counter

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from screening_data import FIXTURE_START, fixture_by_number, load_fixture

from keepout.ccsds.message import UNKNOWN
from keepout.cdm import kvn, make_message_id, xml
from keepout.errors import FleetError
from keepout.pc import computation
from keepout.risk import Action, DecisionConfig, RiskLevel, Thresholds
from keepout.screening import ScreeningVolume
from keepout.synthetic import (
    Fleet,
    FleetSatellite,
    OperatorEvent,
    OperatorResult,
    SyntheticCovarianceModel,
)
from keepout.synthetic.operator import DEFAULT_MODEL, HEADER_COMMENTS, run
from keepout.units import DAY

FIXTURE = load_fixture()
BY_NUMBER = fixture_by_number()
START = FIXTURE_START
FLEET = Fleet(
    (
        FleetSatellite(52939, 5.0, maneuverable=True, name="FLEET ONE"),
        FleetSatellite(44476, 20.0, maneuverable=False),
    ),
    catalog_hard_body_radius=1.0,
)
# The Pc values below are regression snapshots of keepout's own results.


@functools.cache
def fleet_run() -> OperatorResult:
    """:data:`FLEET`'s run over three days, made once: a LEO satellite
    crossing a Starlink at 6 km/s, and Intelsat 39 drifting past the Luch-5X
    inspector satellite at 2 m/s."""
    return run(FLEET, FIXTURE, START, 3 * DAY)


def only(result: OperatorResult, primary: int, secondary: int) -> OperatorEvent:
    (event,) = (
        event
        for event in result.events
        if (event.approach.primary_number, event.approach.secondary_number)
        == (primary, secondary)
    )
    return event


def test_every_approach_becomes_an_event() -> None:
    result = fleet_run()
    approaches = result.screening.approaches
    assert [(a.primary_number, a.secondary_number) for a in approaches] == [
        (52939, 47826),
        (44476, 55841),
    ]
    assert [event.approach for event in result.events] == list(approaches)
    assert result.cdms == tuple(event.event.cdms[0] for event in result.events)
    assert all(len(event.event.cdms) == 1 for event in result.events)
    assert result.fleet is FLEET
    assert result.model is DEFAULT_MODEL
    assert result.created is START
    assert result.screening.start is START
    assert result.screening.duration == 3 * DAY


def test_the_cdm_names_both_objects_and_its_assumptions() -> None:
    result = fleet_run()
    event = only(result, 44476, 55841)
    cdm, tca = event.cdm, event.approach.tca
    assert cdm.version == "2.0"
    assert cdm.originator == "KEEPOUT"
    assert cdm.message_for == "INTELSAT 39 (IS-39)"
    assert cdm.message_id == make_message_id("44476", "55841", tca, START)
    assert cdm.creation_date == START
    assert cdm.header.comments == HEADER_COMMENTS
    assert abs((cdm.tca - tca).to_value("s")) < 1e-6
    fleet, other = cdm.object1, cdm.object2
    assert (fleet.object_id, fleet.name, fleet.international_designator) == (
        "44476",
        "INTELSAT 39 (IS-39)",
        "2019-049B",
    )
    assert (other.object_id, other.name, other.international_designator) == (
        "55841",
        "LUCH-5X (OLYMP-K 2)",
        "2023-031A",
    )
    assert (fleet.hard_body_radius, other.hard_body_radius) == (20.0, 1.0)
    for item, maneuverable, kind in ((fleet, "NO", "PAYLOAD"), (other, "N/A", None)):
        section = item.section
        assert section.get("MANEUVERABLE") == maneuverable
        assert section.get("OBJECT_TYPE") == kind
        assert section.get("COVARIANCE_METHOD") == "DEFAULT"
        assert section.get("EPHEMERIS_NAME") == "NONE"
        assert item.ref_frame == "GCRF"
        omm = BY_NUMBER[int(item.object_id)]
        assert section.comments == DEFAULT_MODEL.comments(omm, tca)
    relative = cdm.relative
    assert relative.get("START_SCREEN_PERIOD") == START
    assert relative.get("STOP_SCREEN_PERIOD") == result.screening.end
    assert relative.get("SCREEN_VOLUME_X") == 5000.0


def test_the_cdm_holds_the_approach_with_the_model_covariances() -> None:
    result = fleet_run()
    for event in result.events:
        approach = event.approach
        conjunction = event.cdm.conjunction()
        pairs = (
            (conjunction.primary, approach.primary_number, approach.primary_state),
            (
                conjunction.secondary,
                approach.secondary_number,
                approach.secondary_state,
            ),
        )
        for item, number, state in pairs:
            assert item.object_id == str(number)
            np.testing.assert_allclose(item.state.r, state.r, rtol=1e-15)
            np.testing.assert_allclose(item.state.v, state.v, rtol=1e-15)
            model = DEFAULT_MODEL.covariance(BY_NUMBER[number], approach.tca)
            assert item.covariance is not None
            np.testing.assert_allclose(
                item.covariance.matrix,
                model.in_gcrf(state).matrix,
                rtol=1e-9,
                atol=1e-9 * model.matrix.max(),
            )
            assert item.covariance.position_only
        satellite = FLEET.find(approach.primary_number)
        assert satellite is not None
        assert conjunction.combined_hard_body_radius == satellite.hard_body_radius + 1.0


def test_each_event_is_assessed_from_its_cdm() -> None:
    result = fleet_run()
    crossing = only(result, 52939, 47826)
    assert crossing.cdm.message_for == "FLEET ONE"
    assert crossing.cdm.object1.section.get("MANEUVERABLE") == "YES"
    assert crossing.observation.source == "keepout (foster)"
    assert crossing.observation.pc == pytest.approx(4.3671e-8, rel=1e-4)
    assert crossing.observation.diagnostics == ()
    assert crossing.assessment.level is RiskLevel.GREEN
    assert crossing.action is Action.IGNORE
    drifting = only(result, 44476, 55841)
    assert drifting.event.event_id == (f"44476/55841/{drifting.cdm.tca.utc.isot}")
    observation = drifting.observation
    assert observation.message_id == drifting.cdm.message_id
    assert observation.time_to_tca == pytest.approx(2.50 * DAY, abs=0.01 * DAY)
    assert observation.pc == pytest.approx(7.3761e-6, rel=1e-4)
    # A slow encounter: the 2-D Pc is kept, with both notes.
    assert len(observation.diagnostics) == 2
    assert "2-D Pc assumptions questionable" in observation.diagnostics[0]
    assert "known only in position" in observation.diagnostics[1]
    assert drifting.assessment.level is RiskLevel.YELLOW
    assert drifting.action is Action.MONITOR
    assert drifting.assessment.trend.observations == (observation,)


def test_fleet_satellites_that_meet_get_an_event_each() -> None:
    pair = Fleet(
        (
            FleetSatellite(62079, 2.0, maneuverable=True),
            FleetSatellite(62080, 3.0, maneuverable=False),
        ),
        catalog_hard_body_radius=1.0,
    )
    result = run(pair, FIXTURE, START, 0.05 * DAY)
    counts = Counter(
        (event.approach.primary_number, event.approach.secondary_number)
        for event in result.events
    )
    assert counts == {(62079, 62080): 2, (62080, 62079): 2}
    first, mirrored = result.events[:2]
    assert first.approach.tca == mirrored.approach.tca
    assert first.cdm.message_for == "SUPERVIEW NEO-2 03"
    other = first.cdm.object2
    assert other.name == "SUPERVIEW NEO-2 04"
    assert other.hard_body_radius == 3.0
    assert other.section.get("MANEUVERABLE") == "NO"
    assert other.section.get("OBJECT_TYPE") == "PAYLOAD"
    assert first.observation.pc == pytest.approx(mirrored.observation.pc, rel=1e-6)


def test_the_options_reach_the_cdms_and_the_assessment() -> None:
    created = START - 1 * u.day
    model = SyntheticCovarianceModel(safety_factor=1.0)
    result = run(
        Fleet((FleetSatellite(52939, 5.0, maneuverable=True),), 1.0),
        FIXTURE,
        START,
        1 * DAY,
        ScreeningVolume.sphere(2_000.0),
        model=model,
        created=created,
        originator="TEST OPERATOR",
        compute=computation("elrod"),
        thresholds=Thresholds(green_below=1e-16, red_from=1e-14),
        decision_config=DecisionConfig(decision_point=2 * DAY),
    )
    (event,) = result.events
    assert result.model is model
    assert result.created is created
    cdm = event.cdm
    assert cdm.originator == "TEST OPERATOR"
    assert cdm.creation_date == created
    assert cdm.relative.get("SCREEN_VOLUME_X") == 2_000.0
    assert "safety factor f 1" in cdm.object1.section.comments[3]
    # Half the default sigmas: a far smaller Pc.
    assert event.observation.source == "keepout (elrod)"
    assert event.observation.pc == pytest.approx(1.5437e-14, rel=1e-4)
    assert event.observation.time_to_tca == pytest.approx(1.817 * DAY, rel=1e-3)
    assert event.assessment.thresholds.red_from == 1e-14
    # Red within the decision point.
    assert event.assessment.level is RiskLevel.RED
    assert event.action is Action.MANEUVER


def test_names_and_designators_the_catalog_lacks_are_unknown() -> None:
    nameless = [
        dataclasses.replace(
            BY_NUMBER[number], object_name="", international_designator=""
        )
        for number in (44476, 55841)
    ]
    fleet = Fleet((FleetSatellite(44476, 20.0, maneuverable=True),), 1.0)
    (event,) = run(fleet, nameless, START, 3 * DAY).events
    cdm = event.cdm
    assert cdm.message_for == UNKNOWN
    for item in (cdm.object1, cdm.object2):
        assert (item.name, item.international_designator) == (UNKNOWN, UNKNOWN)


def test_a_fleet_satellite_needs_an_element_set_in_the_catalog() -> None:
    fleet = Fleet(
        (
            FleetSatellite(25544, 50.0, maneuverable=True),
            FleetSatellite(44476, 20.0, maneuverable=True),
            FleetSatellite(38771, 10.0, maneuverable=True),
        ),
        1.0,
    )
    message = r"no element set in the catalog for fleet satellite\(s\) 25544, 38771$"
    with pytest.raises(FleetError, match=message):
        run(fleet, FIXTURE, START)


def test_the_cdms_are_created_at_one_time() -> None:
    with pytest.raises(ValueError, match="created must be a single time"):
        run(FLEET, FIXTURE, START, created=Time([START, START]))


def test_a_run_gives_the_same_messages_again_with_any_workers() -> None:
    pair = Fleet(
        (
            FleetSatellite(62079, 2.0, maneuverable=True),
            FleetSatellite(52939, 3.0, maneuverable=False),
        ),
        1.0,
    )
    once = run(pair, FIXTURE, START, 0.05 * DAY)
    again = run(pair, reversed(FIXTURE), START, 0.05 * DAY, workers=2)
    assert len(once.cdms) == 2
    assert [kvn.dumps(cdm) for cdm in again.cdms] == [
        kvn.dumps(cdm) for cdm in once.cdms
    ]
    for cdm in once.cdms:
        assert kvn.dumps(xml.loads(xml.dumps(cdm))) == kvn.dumps(cdm)
