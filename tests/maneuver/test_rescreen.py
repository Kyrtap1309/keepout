"""Re-screening a burn: real burns of real orbits against the committed fixture.

Every approach here comes out of :func:`keepout.screening.screen` run on the
fixture, none from a stub. Burns start from ``START``, as the synthetic events'
CDMs are created there:

- 52939 crosses 47826 at 6 km/s, 19.6 h after ``START`` (1029 m, the
  ``CROSSING`` event), and 57087 5.7 days after ``START`` (1082 m). A retrograde
  0.1 m/s takes it out of the event's volume, brings 62118 into its own for the
  first time and moves 57087 by 17 h; 0.03 m/s brings 57087 to 473 m.
- 44476 drifts past 55841 at 2 m/s, 60 h after ``START`` (3647 m, the
  ``DRIFTING`` event). A prograde 0.01 m/s moves that TCA by 3.5 min, and the
  drift brings two more approaches, 36 h before it and a day after.
"""

import dataclasses
import functools
import importlib
import re
from collections.abc import Callable, Sequence
from typing import Any, Final

import astropy.units as u
import pytest
from screening_data import FIXTURE_START, fixture_by_number, load_fixture

from keepout._defaults import TCA_TOLERANCE
from keepout.cdm import CovarianceMethod, ObjectMetadata, make_cdm
from keepout.conjunction import Conjunction
from keepout.errors import GeometryError, ManeuverError
from keepout.events import ConjunctionEvent
from keepout.maneuver import (
    BurnGrid,
    ManeuverOption,
    ManeuverPlan,
    NewApproach,
    RescreenResult,
    judge,
    plan_maneuver,
    rescreen,
)
from keepout.omm import Omm, deduplicate
from keepout.pc import computation
from keepout.propagation import describe_error
from keepout.propagation.maneuvered import ManeuveredPropagator
from keepout.risk import DEFAULT_THRESHOLDS, RiskLevel, Thresholds
from keepout.screening import (
    CloseApproach,
    ScreeningResult,
    ScreeningRole,
    ScreeningVolume,
    SkippedObject,
    SkipReason,
    screen,
)
from keepout.screening.tca import SEARCH_REACH
from keepout.synthetic import Fleet, FleetSatellite, SyntheticCovarianceModel
from keepout.synthetic.operator import DEFAULT_MODEL, run
from keepout.units import DAY, HOUR, elapsed_seconds

FIXTURE: Final = load_fixture()
BY_NUMBER: Final = fixture_by_number()
START: Final = FIXTURE_START
CROSSING: Final = 52939
"""The primary of the crossing event; its secondary is ``CROSSED``."""
CROSSED: Final = 47826
DRIFTING: Final = 44476
"""The primary of the slow event; its secondary is ``PASSED``."""
PASSED: Final = 55841
RECURRING: Final = 57087
"""Approaches 52939 again days after the event."""
CREATED: Final = 62118
"""Comes within 5 km of 52939 only after the retrograde burn."""
FAILING: Final = 45665
"""SGP4 fails for it 41 h 34 min 43 s after ``START``."""
FAILS_AFTER: Final = 149_682.84
"""Seconds from ``START`` to SGP4's first failure for ``FAILING``, to 10 ms."""
UNSUPPORTED: Final = dataclasses.replace(BY_NUMBER[68378], catalog_number=400_000)
"""An element set no SGP4 record holds."""
FLEET: Final = Fleet(
    (
        FleetSatellite(CROSSING, 5.0, maneuverable=True),
        FleetSatellite(DRIFTING, 20.0, maneuverable=True),
    ),
    catalog_hard_body_radius=1.0,
)
REPEATING: Final = 57062
"""Passes ``REPEATED`` twice, 7.8 h after ``START`` at 3401 m (its event) and
48 min later at 671 m."""
REPEATED: Final = 63216
SMALL: Final = BurnGrid(steps_per_orbit=1, delta_v_per_direction=6)
FOSTER: Final = computation("foster")


@functools.cache
def plan_of(primary: int) -> ManeuverPlan:
    """The plan of a synthetic event, created at ``START``."""
    (event,) = (
        event
        for event in run(FLEET, FIXTURE, START, 3 * DAY).events
        if event.approach.primary_number == primary
    )
    return plan_maneuver(event.event, grid=SMALL)


def burn(
    plan: ManeuverPlan, delta_v: float, *, lead: float | None = None
) -> ManeuverOption:
    """A burn of ``plan``, at ``START`` unless a lead is given."""
    return ManeuverOption(
        burn_lead=elapsed_seconds(START, plan.conjunction.tca)
        if lead is None
        else lead,
        delta_v=delta_v,
        pc=0.0,
        range_at_tca=0.0,
        method="foster",
        meets_target=True,
    )


@functools.cache
def rescreened(primary: int, delta_v: float) -> RescreenResult:
    plan = plan_of(primary)
    return rescreen(plan, burn(plan, delta_v), FIXTURE)


def renamed(
    plan: ManeuverPlan, *, primary: str | None = None, secondary: str | None = None
) -> ManeuverPlan:
    """``plan`` with other designators in its latest CDM."""
    old = plan.conjunction
    conjunction = Conjunction(
        dataclasses.replace(old.primary, object_id=primary or old.primary.object_id),
        dataclasses.replace(
            old.secondary, object_id=secondary or old.secondary.object_id
        ),
        old.combined_hard_body_radius,
    )
    cdm = make_cdm(
        conjunction,
        ObjectMetadata("PRI", "2020-001A", covariance_method=CovarianceMethod.DEFAULT),
        ObjectMetadata("SEC", "2019-001A", covariance_method=CovarianceMethod.DEFAULT),
        created=START,
    )
    event = ConjunctionEvent.from_cdms([cdm])
    return dataclasses.replace(plan, event=event, conjunction=conjunction)


def older(number: int, days: float = 40.0) -> Omm:
    """The element set of an object, ``days`` older."""
    omm = BY_NUMBER[number]
    return dataclasses.replace(omm, epoch=omm.epoch - days * u.day)


def without(*numbers: int) -> list[Omm]:
    return [omm for omm in FIXTURE if omm.catalog_number not in numbers]


def by_secondary(result: RescreenResult) -> list[int]:
    return [new.approach.secondary_number for new in result.new_approaches]


def fingerprint(result: RescreenResult) -> list[object]:
    """Everything in a result, exact."""
    new: Sequence[NewApproach] = result.new_approaches
    return [
        result.option,
        result.event_range,
        result.window[0].jd1 + result.window[0].jd2,
        result.window[1].jd1 + result.window[1].jd2,
        result.catalog_objects,
        result.secondary_screened,
        result.skipped_objects,
        [
            (item.approach.secondary_number, item.pc)
            for item in result.closer_approaches
        ],
        [(item.approach.secondary_number, item.pc) for item in result.event_returns],
        [
            (
                item.approach.secondary_number,
                item.approach.tca.jd1,
                item.approach.tca.jd2,
                item.approach.primary_state.r.tolist(),
                item.approach.secondary_state.r.tolist(),
                item.pc,
                item.level,
            )
            for item in new
        ],
    ]


@pytest.fixture
def no_screening(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that screens or integrates anything."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("screened or integrated before the check")

    module = importlib.import_module("keepout.maneuver.rescreen")
    monkeypatch.setattr(module, "screen", refuse)
    monkeypatch.setattr(ManeuveredPropagator, "__init__", refuse)


@pytest.fixture
def no_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that integrates a burn."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("integrated a burn of a primary that was not screened")

    monkeypatch.setattr(ManeuveredPropagator, "__init__", refuse)


def test_zero_burn_has_no_new_approaches() -> None:
    plan = plan_of(CROSSING)
    option = burn(plan, 0.0, lead=6 * HOUR)
    result = rescreen(plan, option, iter(FIXTURE))  # read twice: an iterator must do
    assert result.new_approaches == ()
    assert result.closer_approaches == ()
    assert not result.conflict
    assert result.option is option
    tca = plan.conjunction.tca
    baseline = screen([BY_NUMBER[CROSSING]], FIXTURE, tca - 6 * HOUR * u.s, 7 * DAY)
    (event, _) = baseline.approaches
    assert event.secondary_number == CROSSED
    assert result.event_range == event.range_at_tca
    assert result.secondary_screened
    assert result.catalog_objects == baseline.catalog_count == 217
    # Every object is screened or left out; the primary is among the screened.
    assert result.skipped_objects == len(deduplicate(FIXTURE).kept) - 217
    assert result.skipped_objects > 0
    assert elapsed_seconds(tca - 6 * HOUR * u.s, result.window[0]) == 0.0
    assert elapsed_seconds(result.window[0], result.window[1]) == pytest.approx(7 * DAY)


def test_approach_created_by_the_burn_is_reported() -> None:
    plan = plan_of(CROSSING)
    result = rescreened(CROSSING, -0.1)
    (created, moved) = result.new_approaches
    approach = created.approach
    assert (approach.primary_number, approach.secondary_number) == (CROSSING, CREATED)
    assert approach.tca.isot.startswith("2026-10-12T07:40:16.6")
    assert approach.range_at_tca == pytest.approx(4642.0, abs=1.0)
    assert moved.approach.secondary_number == RECURRING
    # Without the burn, 62118 never comes inside the volume.
    baseline = screen([BY_NUMBER[CROSSING]], FIXTURE, START, 7 * DAY)
    assert CREATED not in {a.secondary_number for a in baseline.approaches}

    conjunction = approach.to_conjunction(
        combined_hard_body_radius=plan.conjunction.combined_hard_body_radius,
        primary_covariance=DEFAULT_MODEL.covariance(BY_NUMBER[CROSSING], approach.tca),
        secondary_covariance=DEFAULT_MODEL.covariance(BY_NUMBER[CREATED], approach.tca),
    )
    assert plan.conjunction.combined_hard_body_radius == 6.0
    assert created.pc == FOSTER(conjunction).value
    assert created.level is RiskLevel.GREEN


def test_new_approaches_use_the_model_given() -> None:
    plan = plan_of(CROSSING)
    wide = SyntheticCovarianceModel(safety_factor=4.0)
    (new,) = rescreen(plan, burn(plan, -0.03), FIXTURE, model=wide).new_approaches
    approach = new.approach
    conjunction = approach.to_conjunction(
        combined_hard_body_radius=plan.conjunction.combined_hard_body_radius,
        primary_covariance=wide.covariance(BY_NUMBER[CROSSING], approach.tca),
        secondary_covariance=wide.covariance(BY_NUMBER[RECURRING], approach.tca),
    )
    assert new.pc == FOSTER(conjunction).value
    assert new.pc != rescreened(CROSSING, -0.03).new_approaches[0].pc


def test_approach_within_tca_tolerance_is_not_new() -> None:
    plan = plan_of(DRIFTING)
    option = burn(plan, 0.01)
    # The event's approach moves, in time and in range: by hand, with the burn.
    baseline = screen([BY_NUMBER[DRIFTING]], FIXTURE, START, 7 * DAY)
    after = screen(
        [BY_NUMBER[DRIFTING]],
        FIXTURE,
        START,
        7 * DAY,
        adjust_primaries=lambda base: ManeuveredPropagator(
            base, START, 0.01, 7 * DAY + SEARCH_REACH
        ),
    )
    (before,) = baseline.approaches
    (_, moved, _) = after.approaches
    shift = elapsed_seconds(before.tca, moved.tca)
    assert 200.0 < shift < TCA_TOLERANCE
    assert moved.range_at_tca != before.range_at_tca

    result = rescreen(plan, option, FIXTURE)
    assert len(result.new_approaches) == 2
    assert all(
        abs(elapsed_seconds(moved.tca, new.approach.tca)) > TCA_TOLERANCE
        for new in result.new_approaches
    )
    assert result.event_range == pytest.approx(moved.range_at_tca, abs=1e-3)


def test_same_secondary_far_in_time_is_new() -> None:
    result = rescreened(DRIFTING, 0.01)
    assert by_secondary(result) == [PASSED, PASSED]
    tca = plan_of(DRIFTING).conjunction.tca
    hours = [
        elapsed_seconds(tca, new.approach.tca) / HOUR for new in result.new_approaches
    ]
    assert hours == [pytest.approx(-36.1, abs=0.1), pytest.approx(24.1, abs=0.1)]


def test_an_approach_of_the_secondary_far_from_the_event_gives_no_event_range() -> None:
    # A retrograde 3 mm/s: the only approach to 55841 is a day before the event's.
    result = rescreened(DRIFTING, -0.003)
    assert by_secondary(result) == [PASSED]
    assert result.event_range is None


def test_another_secondary_at_the_same_time_is_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With a tolerance of 19 h, 57087's approach moved by 17 h is the same one.
    # The baseline's approach of 57087, 14 h from 62118's, says nothing of 62118.
    module = importlib.import_module("keepout.maneuver.rescreen")
    monkeypatch.setattr(module, "TCA_TOLERANCE", 19 * HOUR)
    plan = plan_of(CROSSING)
    assert by_secondary(rescreen(plan, burn(plan, -0.1), FIXTURE)) == [CREATED]


def closer_plan() -> ManeuverPlan:
    """The drifting plan with 47826 as the event's secondary: 55841 is then an
    approach like any other, matched to the baseline."""
    return renamed(plan_of(DRIFTING), secondary=str(CROSSED))


def closer_burn(delta_v: float) -> ManeuverOption:
    return burn(plan_of(DRIFTING), delta_v)


def test_approach_the_burn_brings_closer_is_reported() -> None:
    # 3647 m to 1800 m, the TCA 86 s later: nothing else says so.
    plan = closer_plan()
    (before,) = screen([BY_NUMBER[DRIFTING]], FIXTURE, START, 7 * DAY).approaches
    assert before.secondary_number == PASSED
    result = rescreen(plan, closer_burn(0.003), FIXTURE)
    (closer,) = result.closer_approaches
    approach = closer.approach
    assert approach.secondary_number == PASSED
    assert 0.0 < elapsed_seconds(before.tca, approach.tca) < TCA_TOLERANCE
    assert approach.range_at_tca == pytest.approx(1800.0, abs=1.0)
    assert approach.range_at_tca < before.range_at_tca / 2
    assert result.new_approaches == ()
    # 47826 is the event's secondary here: 55841's range is not the event's.
    assert result.secondary_screened
    assert result.event_range is None
    conjunction = approach.to_conjunction(
        combined_hard_body_radius=plan.conjunction.combined_hard_body_radius,
        primary_covariance=DEFAULT_MODEL.covariance(BY_NUMBER[DRIFTING], approach.tca),
        secondary_covariance=DEFAULT_MODEL.covariance(BY_NUMBER[PASSED], approach.tca),
    )
    assert closer.pc == FOSTER(conjunction).value
    assert closer.level is DEFAULT_THRESHOLDS.level(closer.pc)


def test_matched_approach_that_moved_away_is_not_closer() -> None:
    (before,) = screen([BY_NUMBER[DRIFTING]], FIXTURE, START, 7 * DAY).approaches
    moved = rescreened(DRIFTING, -0.0005)
    assert moved.new_approaches == ()
    assert moved.event_range is not None
    assert moved.event_range > before.range_at_tca
    result = rescreen(closer_plan(), closer_burn(-0.0005), FIXTURE)
    assert result.closer_approaches == ()
    assert result.new_approaches == ()


def repeating_plan() -> ManeuverPlan:
    """The plan of the first of the two approaches of 57062 and 63216."""
    fleet = Fleet(
        (FleetSatellite(REPEATING, 5.0, maneuverable=True),),
        catalog_hard_body_radius=1.0,
    )
    catalog = [BY_NUMBER[REPEATING], BY_NUMBER[REPEATED]]
    (first, _) = run(fleet, catalog, START, 3 * DAY).events
    return plan_maneuver(first.event, grid=SMALL)


def closer_to(result: RescreenResult, secondary: int) -> list[NewApproach]:
    return [
        item
        for item in result.closer_approaches
        if item.approach.secondary_number == secondary
    ]


def test_later_approach_of_the_events_secondary_brought_closer_is_reported() -> None:
    plan = repeating_plan()
    result = rescreen(plan, burn(plan, 0.005), FIXTURE)
    (later,) = closer_to(result, REPEATED)
    away = elapsed_seconds(plan.conjunction.tca, later.approach.tca)
    assert away > TCA_TOLERANCE
    assert later.approach.range_at_tca == pytest.approx(540.0, abs=1.0)
    # The event's own approach moved away, and event_range says so.
    assert result.event_range == pytest.approx(3546.0, abs=1.0)


def test_the_events_own_approach_of_a_repeating_pair_is_not_closer() -> None:
    plan = repeating_plan()
    result = rescreen(plan, burn(plan, -0.005), FIXTURE)
    assert closer_to(result, REPEATED) == []
    assert result.event_range == pytest.approx(3256.0, abs=1.0)


def test_the_events_own_approach_is_not_a_closer_approach() -> None:
    plan = plan_of(DRIFTING)
    result = rescreen(plan, burn(plan, 0.003), FIXTURE)
    assert result.closer_approaches == ()
    assert result.event_range == pytest.approx(1800.0, abs=1.0)


def test_the_events_secondary_back_above_the_target_is_an_event_return() -> None:
    # 55841 comes back 36 h before the event at Pc 1.41e-5, above the target
    # 1e-5, and a day after it at 8.5e-6, below: only the first is a return.
    plan = plan_of(DRIFTING)
    assert plan.target_pc == 1e-5
    result = rescreened(DRIFTING, 0.01)
    early, late = result.new_approaches
    assert early.pc >= plan.target_pc > late.pc
    (back,) = result.event_returns
    assert back is early
    assert back.level is RiskLevel.YELLOW
    assert not result.conflict
    assert not result.clean
    assert elapsed_seconds(back.approach.tca, plan.conjunction.tca) > TCA_TOLERANCE


def test_a_return_below_the_target_is_not_an_event_return() -> None:
    plan = dataclasses.replace(plan_of(DRIFTING), target_pc=2e-5)
    result = rescreen(plan, burn(plan, 0.01), FIXTURE)
    assert len(result.new_approaches) == 2
    assert all(item.pc < plan.target_pc for item in result.new_approaches)
    assert result.event_returns == ()
    assert result.clean


def test_a_return_at_the_target_is_an_event_return() -> None:
    (early, _) = rescreened(DRIFTING, 0.01).new_approaches
    plan = dataclasses.replace(plan_of(DRIFTING), target_pc=early.pc)
    result = rescreen(plan, burn(plan, 0.01), FIXTURE)
    assert [item.pc for item in result.event_returns] == [early.pc]


def test_a_closer_later_approach_of_the_events_secondary_is_an_event_return() -> None:
    # 63216 48 min after the event, brought from 671 m to 540 m: Pc 1.52e-5.
    plan = repeating_plan()
    result = rescreen(plan, burn(plan, 0.005), FIXTURE)
    (later,) = closer_to(result, REPEATED)
    assert later.pc >= plan.target_pc
    assert result.event_returns == (later,)
    assert not result.clean


def test_the_events_own_approach_is_not_an_event_return() -> None:
    # The event's approach, 1800 m after the burn, within the tolerance of its
    # TCA: at any target it is the event_range, never a return.
    plan = dataclasses.replace(plan_of(DRIFTING), target_pc=1e-30)
    result = rescreen(plan, burn(plan, 0.003), FIXTURE)
    assert result.event_range == pytest.approx(1800.0, abs=1.0)
    assert result.event_returns == ()
    assert result.clean


def test_another_objects_approach_is_never_an_event_return() -> None:
    # 57087 at Pc 1.88e-6, above a target of 1e-7 but not the event's secondary.
    plan = dataclasses.replace(plan_of(CROSSING), target_pc=1e-7)
    result = rescreen(plan, burn(plan, -0.03), FIXTURE)
    (new,) = result.new_approaches
    assert new.pc >= plan.target_pc
    assert result.event_returns == ()
    assert result.clean


def test_a_conflict_is_not_clean() -> None:
    plan = plan_of(CROSSING)
    thresholds = Thresholds(green_below=1e-7, red_from=1e-6)
    result = rescreen(plan, burn(plan, -0.03), FIXTURE, thresholds=thresholds)
    assert result.conflict
    assert result.event_returns == ()
    assert not result.clean


def test_closer_red_approach_is_a_conflict() -> None:
    plan = closer_plan()
    plain = rescreen(plan, closer_burn(0.003), FIXTURE)
    (closer,) = plain.closer_approaches
    assert plain.new_approaches == ()
    assert closer.level is not RiskLevel.RED
    assert not plain.conflict
    thresholds = Thresholds(green_below=closer.pc / 10.0, red_from=closer.pc)
    result = rescreen(plan, closer_burn(0.003), FIXTURE, thresholds=thresholds)
    assert result.new_approaches == ()
    assert result.closer_approaches[0].level is RiskLevel.RED
    assert result.conflict


def test_a_closer_approach_is_compared_with_the_baselines_nearest_in_time() -> None:
    (event, later) = screen([BY_NUMBER[CROSSING]], FIXTURE, START, 7 * DAY).approaches
    module = importlib.import_module("keepout.maneuver.rescreen")

    def at(approach: CloseApproach, seconds: float) -> CloseApproach:
        return dataclasses.replace(
            approach,
            secondary_number=CROSSED,
            primary_state=dataclasses.replace(
                approach.primary_state, epoch=event.tca + seconds * u.s
            ),
            secondary_state=dataclasses.replace(
                approach.secondary_state, epoch=event.tca + seconds * u.s
            ),
        )

    # Both within the tolerance of the approach; the nearer is the same one.
    near, far = at(later, 10.0), at(event, -300.0)
    known = dataclasses.replace(
        screen([BY_NUMBER[CROSSING]], [], START, DAY), approaches=(far, near)
    )
    assert module._matched(at(event, 0.0), known) is near
    assert module._matched(at(event, TCA_TOLERANCE + 11.0), known) is None


def counted_screen(
    monkeypatch: pytest.MonkeyPatch,
    second: Callable[[ScreeningResult], ScreeningResult] = lambda result: result,
) -> list[ScreeningResult]:
    """The screenings of ``rescreen``, as they run; ``second`` changes the
    result of the second."""
    module = importlib.import_module("keepout.maneuver.rescreen")
    calls: list[ScreeningResult] = []

    def spy(*args: Any, **kwargs: Any) -> ScreeningResult:
        result = screen(*args, **kwargs)
        calls.append(result)
        return second(result) if len(calls) == 2 else result

    monkeypatch.setattr(module, "screen", spy)
    return calls


def test_primary_left_out_before_the_burn_stops_after_one_screening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = counted_screen(monkeypatch)
    plan = plan_of(CROSSING)
    with pytest.raises(ManeuverError, match=f"primary {CROSSING}.*stale"):
        rescreen(plan, burn(plan, 0.1), [*without(CROSSING), older(CROSSING)])
    assert len(calls) == 1


def test_primary_left_out_only_with_the_burn_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def left_out(result: ScreeningResult) -> ScreeningResult:
        skipped = SkippedObject(
            BY_NUMBER[CROSSING],
            ScreeningRole.PRIMARY,
            SkipReason.PROPAGATION_FAILED,
            "only with the burn",
        )
        return dataclasses.replace(result, skipped=(*result.skipped, skipped))

    calls = counted_screen(monkeypatch, left_out)
    plan = plan_of(CROSSING)
    with pytest.raises(ManeuverError, match=f"primary {CROSSING}.*only with the burn"):
        rescreen(plan, burn(plan, 0.0), FIXTURE)
    assert len(calls) == 2


def test_new_red_approach_is_a_conflict() -> None:
    plan = plan_of(CROSSING)
    # 57087 at 473 m, 4.7 h after the approach without the burn: Pc 1.88e-6.
    thresholds = Thresholds(green_below=1e-7, red_from=1e-6)
    result = rescreen(plan, burn(plan, -0.03), FIXTURE, thresholds=thresholds)
    (new,) = result.new_approaches
    assert new.approach.secondary_number == RECURRING
    assert new.approach.range_at_tca == pytest.approx(473.0, abs=1.0)
    assert new.pc == pytest.approx(1.878e-6, rel=1e-3)
    assert new.level is RiskLevel.RED
    assert result.conflict


def test_new_yellow_approach_is_not_a_conflict() -> None:
    result = rescreened(CROSSING, -0.03)
    (new,) = result.new_approaches
    assert new.level is RiskLevel.YELLOW
    assert not result.conflict


def test_event_range_none_when_outside_the_volume() -> None:
    result = rescreened(CROSSING, -0.1)
    assert result.event_range is None
    assert CROSSED not in by_secondary(result)


def test_event_range_follows_the_burn() -> None:
    assert rescreened(CROSSING, 0.01).event_range == pytest.approx(1838.0, abs=1.0)


def test_volume_and_duration_are_those_given() -> None:
    plan = plan_of(CROSSING)
    option = burn(plan, 0.0)
    small = ScreeningVolume.sphere(1000.0)
    assert rescreen(plan, option, FIXTURE, volume=small).event_range is None
    result = rescreen(plan, burn(plan, -0.1), FIXTURE, duration=DAY)
    assert result.new_approaches == ()
    assert elapsed_seconds(result.window[0], result.window[1]) == pytest.approx(DAY)


def test_older_element_sets_are_not_a_skipped_primary() -> None:
    plan = plan_of(CROSSING)
    catalog = [*FIXTURE, older(CROSSING), older(CROSSED)]
    result = rescreen(plan, burn(plan, 0.0), catalog)
    assert result.new_approaches == ()
    assert result.event_range == pytest.approx(1029.0, abs=1.0)
    assert result.secondary_screened
    # Older sets of objects that are screened leave no object out.
    assert (
        result.skipped_objects
        == rescreen(plan, burn(plan, 0.0), FIXTURE).skipped_objects
    )


def test_designators_are_compared_as_catalog_numbers() -> None:
    plain = plan_of(CROSSING)
    plan = renamed(plain, primary="052939", secondary="0047826")
    result = rescreen(plan, burn(plain, 0.0), FIXTURE)
    assert result.event_range == pytest.approx(1029.0, abs=1.0)
    assert result.new_approaches == ()


@pytest.mark.usefixtures("no_screening")
def test_primary_missing_from_catalog_is_an_error() -> None:
    plan = plan_of(CROSSING)
    with pytest.raises(ManeuverError, match=f"primary {CROSSING}.*no element set"):
        rescreen(plan, burn(plan, 0.1), without(CROSSING))


@pytest.mark.usefixtures("no_screening")
@pytest.mark.parametrize("designator", ["2020-001A", "52_939", "+52939", "٥٢٩٣٩"])
def test_designator_that_is_no_catalog_number_is_an_error(designator: str) -> None:
    plain = plan_of(CROSSING)
    plan = renamed(plain, primary=designator)
    with pytest.raises(
        ManeuverError, match=rf"primary {re.escape(designator)}.*catalog number"
    ):
        rescreen(plan, burn(plain, 0.1), FIXTURE)


def test_secondary_missing_from_catalog_leaves_the_rest_screened() -> None:
    # An analyst object, say: not in a public catalog, and not "outside".
    plan = plan_of(CROSSING)
    result = rescreen(plan, burn(plan, -0.1), without(CROSSED))
    assert not result.secondary_screened
    assert result.event_range is None
    assert by_secondary(result) == [CREATED, RECURRING]
    # It was never given, so screening left nothing out.
    assert result.skipped_objects == rescreened(CROSSING, -0.1).skipped_objects


@pytest.mark.parametrize("designator", ["2019-001A", "47_826", "+47826"])
def test_secondary_designator_that_is_no_catalog_number_is_not_screened(
    designator: str,
) -> None:
    plain = plan_of(CROSSING)
    plan = renamed(plain, secondary=designator)
    # 1838 m from 47826 with this burn, if the designator were taken for it.
    result = rescreen(plan, burn(plain, 0.01), FIXTURE)
    assert not result.secondary_screened
    assert result.event_range is None
    assert by_secondary(result) == [RECURRING]


@pytest.mark.usefixtures("no_integration")
def test_skipped_primary_is_an_error() -> None:
    plan = plan_of(CROSSING)
    catalog = [*without(CROSSING), older(CROSSING)]
    with pytest.raises(ManeuverError, match=f"primary {CROSSING}.*stale"):
        rescreen(plan, burn(plan, 0.1), catalog)


@pytest.mark.usefixtures("no_integration")
def test_primary_sgp4_cannot_take_is_an_error() -> None:
    plain = plan_of(CROSSING)
    plan = renamed(plain, primary="400000")
    with pytest.raises(ManeuverError, match=r"primary 400000.*unsupported"):
        rescreen(plan, burn(plain, 0.1), [*FIXTURE, UNSUPPORTED])


@pytest.mark.usefixtures("no_integration")
def test_primary_without_a_volume_is_an_error() -> None:
    plan = plan_of(CROSSING)

    def refusing(primary: Omm) -> ScreeningVolume:
        raise GeometryError(f"no volume for {primary.catalog_number}")

    with pytest.raises(ManeuverError, match=f"primary {CROSSING}.*no volume"):
        rescreen(plan, burn(plan, 0.1), FIXTURE, volume=refusing)


@pytest.mark.usefixtures("no_integration")
def test_primary_sgp4_fails_for_in_the_window_is_an_error() -> None:
    plain = plan_of(CROSSING)
    plan = renamed(plain, primary=str(FAILING))
    with pytest.raises(
        ManeuverError, match=f"primary {FAILING}.*propagation failed.*2026-10-07"
    ):
        rescreen(plan, burn(plain, 0.0), FIXTURE)


def test_skipped_secondary_leaves_the_rest_screened() -> None:
    plan = plan_of(CROSSING)
    catalog = [*without(CROSSED), older(CROSSED)]
    result = rescreen(plan, burn(plan, -0.1), catalog)
    assert not result.secondary_screened
    assert result.event_range is None
    assert by_secondary(result) == [CREATED, RECURRING]
    plain = rescreened(CROSSING, -0.1)
    assert result.skipped_objects == plain.skipped_objects + 1
    assert result.catalog_objects == plain.catalog_objects - 1


def test_secondary_seen_failing_only_after_the_burn_is_not_screened() -> None:
    # Without the burn the filter rules 45665 out of the pair, so nobody finds
    # out that SGP4 fails for it; the burn widens the primary's radius bounds.
    plain = plan_of(CROSSING)
    plan = renamed(plain, secondary=str(FAILING))
    baseline = screen([BY_NUMBER[CROSSING]], [BY_NUMBER[FAILING]], START, 7 * DAY)
    assert baseline.skipped == ()
    still = rescreen(plan, burn(plain, 0.0), FIXTURE)
    assert still.secondary_screened
    result = rescreen(plan, burn(plain, 1.0), FIXTURE)
    assert not result.secondary_screened
    assert result.event_range is None
    # Counted from the screening with the burn, as the other objects it failed.
    assert result.skipped_objects > still.skipped_objects
    unique = len(deduplicate(FIXTURE).kept)
    assert result.catalog_objects + result.skipped_objects == unique


@pytest.mark.usefixtures("no_screening")
@pytest.mark.parametrize("delta_v", [0.0, 0.001])
def test_primary_sgp4_cannot_carry_through_the_window_is_an_error(
    delta_v: float,
) -> None:
    # The two orbits' radii keep the pair apart, so screening never propagates
    # 45665, and its failure 41 h 34 min 43 s after START would read as "nothing
    # new".
    plain = plan_of(CROSSING)
    plan = renamed(plain, primary=str(FAILING))
    catalog = [BY_NUMBER[FAILING], BY_NUMBER[CROSSING]]
    with pytest.raises(
        ManeuverError,
        match=rf"primary {FAILING}.*propagation failed: "
        rf"{re.escape(describe_error(1))}, at 2026-10-07T17:3",
    ):
        rescreen(plan, burn(plain, delta_v), catalog)


def test_screening_alone_does_not_see_that_primary_fail() -> None:
    # Why rescreen looks for itself: the failure leaves no trace in the result.
    catalog = [BY_NUMBER[FAILING], BY_NUMBER[CROSSING]]
    result = screen([BY_NUMBER[FAILING]], catalog, START, 7 * DAY)
    assert result.skipped == ()
    assert FAILING in result.volumes
    assert result.approaches == ()


def thin_plan() -> tuple[ManeuverPlan, ManeuverOption, list[Omm]]:
    """The plan of ``FAILING`` as primary, with a small burn and a catalog of
    itself and 52939."""
    plain = plan_of(CROSSING)
    catalog = [BY_NUMBER[FAILING], BY_NUMBER[CROSSING]]
    return renamed(plain, primary=str(FAILING)), burn(plain, 0.001), catalog


def test_primary_failing_after_the_window_is_screened() -> None:
    plan, option, catalog = thin_plan()
    last = FAILS_AFTER - 2.84  # the search reads 2 s further
    result = rescreen(plan, option, catalog, duration=last)
    assert result.new_approaches == ()


@pytest.mark.usefixtures("no_screening")
def test_the_end_of_the_window_is_checked() -> None:
    # Between grid times: no time on the 60 s grid fails, only the end does.
    plan, option, catalog = thin_plan()
    duration = FAILS_AFTER + 7.16
    assert duration % 60.0 > FAILS_AFTER % 60.0
    assert duration // 60.0 == FAILS_AFTER // 60.0
    with pytest.raises(ManeuverError, match=r"propagation failed.*17:34:5"):
        rescreen(plan, option, catalog, duration=duration)


def test_burn_into_the_earth_is_an_error_naming_the_burn() -> None:
    plan = plan_of(CROSSING)
    with pytest.raises(ManeuverError, match=r"burn of -1000 m/s 19\.6 h before TCA"):
        rescreen(plan, burn(plan, -1000.0), FIXTURE)


@pytest.mark.usefixtures("no_screening")
@pytest.mark.parametrize(
    ("lead", "duration"),
    [
        (0.0, 7 * DAY),
        (TCA_TOLERANCE - 1.0, 7 * DAY),
        (8 * DAY, 7 * DAY),
        (7 * DAY, 7 * DAY),
        (7 * DAY - TCA_TOLERANCE + 1.0, 7 * DAY),
    ],
)
def test_window_without_the_event_and_its_tolerance_is_an_error(
    lead: float, duration: float
) -> None:
    plan = plan_of(CROSSING)
    option = burn(plan, 0.1, lead=lead)
    with pytest.raises(ManeuverError, match="does not hold the event's TCA"):
        rescreen(plan, option, FIXTURE, duration=duration)


def test_window_ending_at_the_event_would_lose_a_later_tca() -> None:
    # A prograde 1 mm/s moves this TCA later: a window ending at the event's
    # TCA has no minimum, though the secondary is inside the volume.
    plan = plan_of(CROSSING)
    lead = elapsed_seconds(START, plan.conjunction.tca)
    option = burn(plan, 0.001)
    with pytest.raises(ManeuverError, match="does not hold the event's TCA"):
        rescreen(plan, option, FIXTURE, duration=lead)
    result = rescreen(plan, option, FIXTURE, duration=lead + TCA_TOLERANCE)
    assert result.event_range is not None


@pytest.mark.parametrize(
    ("lead", "duration"),
    [(TCA_TOLERANCE, DAY), (6 * HOUR, 6 * HOUR + TCA_TOLERANCE)],
)
def test_window_just_holding_the_event_and_its_tolerance_is_screened(
    lead: float, duration: float
) -> None:
    plan = plan_of(CROSSING)
    result = rescreen(plan, burn(plan, 0.0, lead=lead), FIXTURE, duration=duration)
    assert result.event_range == pytest.approx(1029.0, abs=1.0)
    assert result.secondary_screened


@pytest.mark.usefixtures("no_screening")
def test_negative_duration_is_refused() -> None:
    plan = plan_of(CROSSING)
    with pytest.raises(ValueError, match="duration"):
        rescreen(plan, burn(plan, 0.1), FIXTURE, duration=-1.0)


def test_result_does_not_depend_on_workers() -> None:
    plan = plan_of(CROSSING)
    together = rescreened(CROSSING, -0.1)
    apart = rescreen(plan, burn(plan, -0.1), FIXTURE, workers=2)
    assert fingerprint(apart) == fingerprint(together)


def judged(*, red: bool = False, back: bool = False) -> RescreenResult:
    """A re-screen of a burn whose one new approach is red or green, and
    which brings the event's pair back or not."""
    found = rescreened(CROSSING, -0.03)
    (new,) = found.new_approaches
    level = RiskLevel.RED if red else RiskLevel.GREEN
    return dataclasses.replace(
        found,
        new_approaches=(dataclasses.replace(new, level=level),),
        event_returns=(new,) if back else (),
    )


def test_no_burn_cannot_be_recommended() -> None:
    verdict = judge([])
    assert (verdict.burns, verdict.recommendable) == (0, False)


def test_a_burn_missing_the_target_cannot_be_recommended() -> None:
    plan = plan_of(CROSSING)
    missing = dataclasses.replace(burn(plan, -0.03), meets_target=False)
    verdict = judge([(missing, None), (burn(plan, 0.03), judged())])
    assert verdict.missed_target
    assert not verdict.recommendable


@pytest.mark.parametrize("other", [None, "clean"])
def test_one_clean_or_unscreened_burn_is_enough(other: str | None) -> None:
    plan = plan_of(CROSSING)
    second = None if other is None else judged()
    verdict = judge(
        [(burn(plan, -0.03), judged(red=True, back=True)), (burn(plan, 0.03), second)]
    )
    assert verdict.recommendable
    assert not verdict.brings_red
    assert not verdict.brings_pair_back


def test_every_burn_with_a_red_approach_or_a_return_cannot_be_recommended() -> None:
    plan = plan_of(CROSSING)
    verdict = judge(
        [
            (burn(plan, -0.03), judged(red=True)),
            (burn(plan, 0.03), judged(back=True)),
        ]
    )
    assert (verdict.burns, verdict.brings_red, verdict.brings_pair_back) == (
        2,
        True,
        True,
    )
    assert not verdict.missed_target
    assert not verdict.recommendable


def test_new_approaches_take_the_pc_method_and_basis_given() -> None:
    plan = plan_of(CROSSING)
    (new,) = rescreened(CROSSING, -0.03).new_approaches
    approach = new.approach
    conjunction = approach.to_conjunction(
        combined_hard_body_radius=plan.conjunction.combined_hard_body_radius,
        primary_covariance=DEFAULT_MODEL.covariance(BY_NUMBER[CROSSING], approach.tca),
        secondary_covariance=DEFAULT_MODEL.covariance(
            BY_NUMBER[RECURRING], approach.tca
        ),
    )
    # A Monte Carlo interval whose upper end the conservative basis takes.
    compute = computation("montecarlo", samples=20_000, seed=1)
    (conservative,) = rescreen(
        plan, burn(plan, -0.03), FIXTURE, compute=compute
    ).new_approaches
    upper = compute(conjunction).interval
    assert upper is not None
    assert conservative.pc == upper[1]
