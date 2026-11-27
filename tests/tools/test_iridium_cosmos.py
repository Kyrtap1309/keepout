"""The Iridium-Cosmos case study's computations, on the committed excerpt."""

import astropy.units as u
import iridium_cosmos as study
import pytest

from keepout.events import DEFAULT_TCA_TOLERANCE, ConjunctionEvent
from keepout.omm import Omm
from keepout.risk import DEFAULT_THRESHOLDS, Assessment
from keepout.units import DAY, HOUR


@pytest.fixture(scope="module")
def series() -> tuple[study.Issue, ...]:
    return study.cdm_series(study.read_pair())


@pytest.fixture(scope="module")
def event(series: tuple[study.Issue, ...]) -> ConjunctionEvent:
    return study.collision_event(series)


@pytest.fixture(scope="module")
def assessments(event: ConjunctionEvent) -> tuple[Assessment, Assessment]:
    return study.assessments(event)


def test_series_covers_seven_days(series: tuple[study.Issue, ...]) -> None:
    start = study.TCA_REFERENCE - study.SERIES_SPAN * u.s
    times = [issue.time for issue in series]
    assert times == sorted(times)
    assert all(start <= time < study.TCA_REFERENCE for time in times)
    pair_epochs = {
        omm.epoch.isot
        for omm in study.read_pair()
        if start <= omm.epoch < study.TCA_REFERENCE
    }
    assert {time.isot for time in times} == pair_epochs


def test_series_tca_near_reference(series: tuple[study.Issue, ...]) -> None:
    found = [issue.cdm for issue in series if issue.cdm is not None]
    assert len(found) >= 10
    for cdm in found:
        assert (
            abs((cdm.tca - study.TCA_REFERENCE).to_value(u.s)) <= DEFAULT_TCA_TOLERANCE
        )
        assert (cdm.object1.object_id, cdm.object2.object_id) == ("24946", "22675")
    issues = {issue.time.isot: issue for issue in series}
    for cdm in found:
        assert issues[cdm.creation_date.isot].cdm is cdm


def test_issue_without_secondary_listed() -> None:
    pair = study.read_pair()
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
        radii = [cdm.object1.hard_body_radius, cdm.object2.hard_body_radius]
        assert sum(r or 0.0 for r in radii) == pytest.approx(study.BASE_HBR)


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


def test_series_with_other_radius_changes_only_radius() -> None:
    pair = study.read_pair()
    base = [i.cdm for i in study.cdm_series(pair) if i.cdm is not None]
    wider = [i.cdm for i in study.cdm_series(pair, hbr=10.0) if i.cdm is not None]
    assert [c.miss_distance for c in base] == [c.miss_distance for c in wider]
    for cdm in wider:
        radii = [cdm.object1.hard_body_radius, cdm.object2.hard_body_radius]
        assert sum(r or 0.0 for r in radii) == pytest.approx(10.0)


@pytest.fixture(scope="module")
def excerpt() -> tuple[Omm, ...]:
    return study.read_catalog()


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
        cdm = event.cdm
        radii = [cdm.object1.hard_body_radius, cdm.object2.hard_body_radius]
        assert sum(r or 0.0 for r in radii) == pytest.approx(study.BASE_HBR)


@pytest.fixture(scope="module")
def maneuver(event: ConjunctionEvent, excerpt: tuple[Omm, ...]) -> study.Maneuver:
    return study.maneuver(event, excerpt)


@pytest.mark.slow
def test_burn_grid_window(maneuver: study.Maneuver) -> None:
    leads = maneuver.plan.burn_leads
    assert leads.max() <= 2 * DAY
    assert leads.min() >= 2 * HOUR
    assert maneuver.plan.event.latest.creation_date <= study.IRIDIUM_AS_OF
    assert set(maneuver.rescreens) == {o.direction for o in maneuver.plan.options}
    assert set(maneuver.shifts) == set(maneuver.rescreens)
    assert maneuver.target == DEFAULT_THRESHOLDS.green_below
    for option in maneuver.plan.options:
        assert option.pc < DEFAULT_THRESHOLDS.green_below


@pytest.mark.slow
def test_shift_beyond_excerpt_fails(
    maneuver: study.Maneuver, monkeypatch: pytest.MonkeyPatch
) -> None:
    study.require_within_excerpt(maneuver.shifts)
    monkeypatch.setattr(study, "EXCERPT_RADIUS", 1.0)
    with pytest.raises(study.ExcerptTooSmall, match="1 m"):
        study.require_within_excerpt(maneuver.shifts)
