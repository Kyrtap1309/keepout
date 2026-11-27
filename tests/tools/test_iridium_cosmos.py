"""The Iridium-Cosmos case study's computations, on the committed excerpt."""

import astropy.units as u
import iridium_cosmos as study
import iridium_cosmos_data
import pytest
from astropy.time import Time
from screening_data import load_fixture

from keepout.cdm import Cdm
from keepout.events import DEFAULT_TCA_TOLERANCE, ConjunctionEvent
from keepout.omm import Omm
from keepout.risk import Assessment
from keepout.screening import DEFAULT_DURATION
from keepout.units import DAY, HOUR, KM


def combined_hbr(cdm: Cdm) -> float:
    """The CDM's combined hard-body radius; both objects must state theirs."""
    radii = (cdm.object1.hard_body_radius, cdm.object2.hard_body_radius)
    assert None not in radii
    return sum(r for r in radii if r is not None)


@pytest.fixture(scope="module")
def excerpt() -> tuple[Omm, ...]:
    return study.read_catalog()


@pytest.fixture(scope="module")
def pair(excerpt: tuple[Omm, ...]) -> tuple[Omm, ...]:
    return study.pair_of(excerpt)


@pytest.fixture(scope="module")
def selection() -> iridium_cosmos_data.Selection:
    return iridium_cosmos_data.read_manifest(study.DATA)["selection"]


@pytest.fixture(scope="module")
def series(pair: tuple[Omm, ...]) -> tuple[study.Issue, ...]:
    return study.cdm_series(pair)


@pytest.fixture(scope="module")
def event(series: tuple[study.Issue, ...]) -> ConjunctionEvent:
    return study.collision_event(series)


@pytest.fixture(scope="module")
def assessments(event: ConjunctionEvent) -> tuple[Assessment, Assessment]:
    return study.assessments(event)


def test_series_covers_seven_days(
    series: tuple[study.Issue, ...], pair: tuple[Omm, ...]
) -> None:
    start = study.TCA_REFERENCE - study.SERIES_SPAN * u.s
    times = [issue.time for issue in series]
    assert times == sorted(times)
    assert all(start <= time < study.TCA_REFERENCE for time in times)
    pair_epochs = {
        omm.epoch.isot for omm in pair if start <= omm.epoch < study.TCA_REFERENCE
    }
    assert {time.isot for time in times} == pair_epochs


def test_series_tca_near_reference(series: tuple[study.Issue, ...]) -> None:
    found = [issue.cdm for issue in series if issue.cdm is not None]
    assert len(found) >= 10
    for cdm in found:
        assert (
            abs((cdm.tca - study.TCA_REFERENCE).to_value(u.s)) <= DEFAULT_TCA_TOLERANCE
        )
        assert (cdm.object1.object_id, cdm.object2.object_id) == (
            str(study.IRIDIUM_33),
            str(study.COSMOS_2251),
        )


def test_each_cdm_is_created_at_its_issue(series: tuple[study.Issue, ...]) -> None:
    issues = {issue.time.isot: issue for issue in series}
    for issue in series:
        if issue.cdm is not None:
            assert issues[issue.cdm.creation_date.isot].cdm is issue.cdm


def test_issue_without_secondary_listed(pair: tuple[Omm, ...]) -> None:
    first_cosmos = study.TCA_REFERENCE - 3 * u.day
    late = tuple(
        omm
        for omm in pair
        if omm.catalog_number == study.IRIDIUM_33 or omm.epoch >= first_cosmos
    )
    series = study.cdm_series(late)
    early = [issue for issue in series if issue.time < first_cosmos]
    assert early
    for issue in early:
        assert issue.cdm is None
        assert issue.reason == "no element set of 22675 known"


def test_cdms_state_the_model_and_radius(event: ConjunctionEvent) -> None:
    for cdm in event.cdms:
        assert cdm.assumed_covariance
        assert combined_hbr(cdm) == pytest.approx(study.BASE_HBR)


def test_decision_uses_only_known_cdms(
    assessments: tuple[Assessment, Assessment], event: ConjunctionEvent
) -> None:
    at_decision, final = assessments
    known = {
        cdm.message_id for cdm in event.cdms if cdm.creation_date <= study.IRIDIUM_AS_OF
    }
    assert {o.message_id for o in at_decision.trend.observations} == known
    assert len(final.trend.observations) == len(event.cdms)
    assert len(known) < len(event.cdms)


def test_max_pc_bounds_model_pc(event: ConjunctionEvent) -> None:
    observations = event.observations()
    for cdm, observation in zip(event.cdms, observations, strict=True):
        assert observation.pc is not None
        bounds = study.max_pcs(cdm, study.BASE_HBR)
        assert set(bounds) == {"frisbee", "dilution", "alfano"}
        for value in bounds.values():
            assert value >= observation.pc * (1 - 1e-9)


def test_hbr_sensitivity_rises_with_radius(event: ConjunctionEvent) -> None:
    table = study.hbr_sensitivity(event)
    assert list(table) == sorted({study.BASE_HBR, *study.HBR_SENSITIVITY})
    for column in (0, 1):
        values = [table[radius][column] for radius in table]
        assert values == sorted(values)
        assert values[0] < values[-1]


def test_series_with_other_radius_changes_only_radius(
    series: tuple[study.Issue, ...], pair: tuple[Omm, ...]
) -> None:
    base = [i.cdm for i in series if i.cdm is not None]
    wider = [i.cdm for i in study.cdm_series(pair, hbr=10.0) if i.cdm is not None]
    assert [c.miss_distance for c in base] == [c.miss_distance for c in wider]
    for cdm in wider:
        assert combined_hbr(cdm) == pytest.approx(10.0)


@pytest.fixture(scope="module")
def ranked(excerpt: tuple[Omm, ...]) -> study.Ranked:
    return study.rank(excerpt)


@pytest.mark.slow
def test_rank_finds_collision(ranked: study.Ranked) -> None:
    assert ranked.collision.approach.secondary_number == study.COSMOS_2251
    assert 1 <= ranked.by_pc <= ranked.approaches
    assert 1 <= ranked.by_miss <= ranked.approaches
    assert ranked.approaches == len(ranked.result.events)
    assert ranked.result.created == study.IRIDIUM_AS_OF


@pytest.mark.slow
def test_rank_same_hbr_for_all(ranked: study.Ranked) -> None:
    for event in ranked.result.events:
        assert combined_hbr(event.cdm) == pytest.approx(study.BASE_HBR)


@pytest.mark.slow
def test_rank_orders_events_by_pc(ranked: study.Ranked) -> None:
    pcs = [event.observation.pc for event in ranked.events_by_pc]
    known = [pc for pc in pcs if pc is not None]
    assert known == sorted(known, reverse=True)
    assert pcs[: len(known)] == known


@pytest.fixture(scope="module")
def maneuver(
    event: ConjunctionEvent,
    excerpt: tuple[Omm, ...],
    selection: iridium_cosmos_data.Selection,
) -> study.Maneuver:
    return study.maneuver(event, excerpt, selection)


@pytest.mark.slow
def test_burn_grid_window(maneuver: study.Maneuver) -> None:
    leads = maneuver.plan.burn_leads
    assert leads.max() <= 2 * DAY
    assert leads.min() >= 2 * HOUR
    assert maneuver.plan.event.latest.creation_date <= study.IRIDIUM_AS_OF
    assert set(maneuver.rescreens) == {o.direction for o in maneuver.plan.options}
    assert set(maneuver.shifts) == set(maneuver.rescreens)
    for option in maneuver.plan.options:
        assert option.pc < study.MANEUVER_TARGET


@pytest.mark.slow
def test_shift_beyond_excerpt_fails(
    maneuver: study.Maneuver, selection: iridium_cosmos_data.Selection
) -> None:
    windows = [result.window for result in maneuver.rescreens.values()]
    study.require_within_excerpt(maneuver.shifts, windows, selection)
    small: iridium_cosmos_data.Selection = {**selection, "radius_km": 1.0}
    with pytest.raises(study.ExcerptTooSmall, match="1000 m"):
        study.require_within_excerpt(maneuver.shifts, windows, small)


def test_rescreen_window_beyond_excerpt_fails(
    selection: iridium_cosmos_data.Selection,
) -> None:
    start = Time(selection["screening_start"], scale="utc")
    end = start + selection["screening_days"] * u.day
    study.require_within_excerpt({}, [(start, end)], selection)
    late = (start + 2 * u.hour, end + 2 * u.hour)
    with pytest.raises(study.ExcerptTooSmall, match="after the excerpt's screening"):
        study.require_within_excerpt({}, [late], selection)


def test_excerpt_covers_the_latest_burn_s_week(
    selection: iridium_cosmos_data.Selection,
) -> None:
    assert selection["radius_km"] * KM == study.EXCERPT_RADIUS
    start = Time(selection["screening_start"], scale="utc")
    covered = start + selection["screening_days"] * u.day
    latest_burn = study.TCA_REFERENCE - study.BURN_GRID.latest_lead * u.s
    assert covered + 1 * u.s >= latest_burn + DEFAULT_DURATION * u.s


@pytest.fixture(scope="module")
def fixture_catalog() -> tuple[Omm, ...]:
    return load_fixture()


def test_debris_counted_by_parent(fixture_catalog: tuple[Omm, ...]) -> None:
    start = max(omm.epoch for omm in fixture_catalog)
    today = study.debris_today(fixture_catalog, start, workers=1)
    # The fixture holds 2 Iridium 33 and 6 Cosmos 2251 fragments.
    assert today.on_orbit == {"IRIDIUM 33": 2, "COSMOS 2251": 6}
    ranges = [approach.range_at_tca for approach in today.closest]
    assert ranges == sorted(ranges)
    assert len(today.closest) <= study.DEBRIS_CLOSEST
    pieces = {omm.catalog_number for omm in fixture_catalog if study.is_debris(omm)}
    for approach in today.closest:
        assert approach.primary_number in pieces
        assert approach.secondary_number not in pieces
    assert today.approaches >= len(today.closest)


def test_launch_pieces_are_not_debris(pair: tuple[Omm, ...]) -> None:
    omm = next(o for o in pair if o.catalog_number == study.IRIDIUM_33)
    assert not study.is_debris(omm)
