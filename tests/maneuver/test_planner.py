"""The Pc grid of a conjunction event over Δv and burn time, and its options."""

import dataclasses
import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, TypedDict

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from builders import (
    CROSSING_INCLINATION,
    CROSSING_MISS,
    CROSSING_RADIUS,
    CROSSING_SIGMA,
    CROSSING_TCA,
    SMALL_GRID,
    approx,
    crossing_conjunction,
)
from faults import (
    auto_hall_integration,
    auto_monte_carlo_budget,
    hall3d_failing,
    recorded_runs,
)
from scipy.stats import ncx2

from keepout._defaults import DECISION_POINT
from keepout._types import FloatArray, override
from keepout.cdm import (
    Cdm,
    CovarianceMethod,
    Maneuverable,
    ObjectMetadata,
    make_cdm,
)
from keepout.cdm.model import Field, Section
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.errors import (
    HardBodyRadiusError,
    ManeuverError,
    PcConvergenceError,
    PropagationError,
)
from keepout.events import ConjunctionEvent
from keepout.maneuver import (
    DEFAULT_BURN_GRID,
    MAX_CONFIRMATIONS,
    MIN_BURN_LEAD,
    TARGET_PC,
    BurnDirection,
    BurnEffects,
    BurnGrid,
    ManeuverOption,
    ManeuverPlan,
    burn_effects,
    evaluate_burn,
    plan_maneuver,
    planner,
    prepare_maneuver,
)
from keepout.objects import ObjectState
from keepout.orbits import EARTH_MU, orbital_period
from keepout.pc import (
    PcComputation,
    PcResult,
    computation,
    compute_pc,
)
from keepout.pc.api import AUTO_MC_SEED
from keepout.propagation.forces import EARTH_RADIUS, ForceModel
from keepout.risk import PC_FLOOR, Thresholds
from keepout.state import StateVector
from keepout.units import HOUR, KM

TCA: Final = CROSSING_TCA
RADIUS: Final = CROSSING_RADIUS
SPEED: Final = math.sqrt(EARTH_MU / RADIUS)
INCLINATION: Final = CROSSING_INCLINATION
SIGMA: Final = CROSSING_SIGMA
"""Position sigma of each object in :func:`crossing_conjunction` [m]."""
MISS: Final = CROSSING_MISS
"""Radial miss of :func:`crossing_conjunction` [m]."""
FOSTER: Final = computation("foster")


def direction(angle: float) -> FloatArray:
    unit: FloatArray = np.array([0.0, math.cos(angle), math.sin(angle)])
    return unit


def item(
    object_id: str,
    r: FloatArray,
    v: FloatArray,
    sigma: float,
    *,
    velocity_sigma: float,
) -> ObjectState:
    matrix = np.diag([sigma**2] * 3 + [velocity_sigma**2] * 3)
    return ObjectState(
        object_id,
        StateVector(TCA, r, v, "GCRF"),
        Covariance(matrix, "GCRF", TCA),
        10.0,
    )


def as_cdm(
    conjunction: Conjunction,
    *,
    hours: float = 25.0,
    maneuverable: Maneuverable = Maneuverable.YES,
    method: CovarianceMethod = CovarianceMethod.CALCULATED,
) -> Cdm:
    return make_cdm(
        conjunction,
        ObjectMetadata(
            "PRI",
            "2020-001A",
            maneuverable=maneuverable,
            covariance_method=method,
        ),
        ObjectMetadata("SEC", "2019-001A", covariance_method=method),
        created=TCA - hours * HOUR * u.s,
    )


def slow_conjunction(*, velocity_sigma: float = 0.01) -> Conjunction:
    """Nearly co-orbital objects passing at 2 m/s, 50 m apart."""
    along = direction(INCLINATION)
    across = np.cross([1.0, 0.0, 0.0], along)
    primary = item(
        "1001",
        np.array([RADIUS, 0.0, 0.0]),
        SPEED * along,
        SIGMA,
        velocity_sigma=velocity_sigma,
    )
    secondary = item(
        "2002",
        np.array([RADIUS + 50.0, 0.0, 0.0]),
        SPEED * along + 2.0 * across,
        SIGMA,
        velocity_sigma=velocity_sigma,
    )
    return Conjunction(primary, secondary, 20.0)


def event_of(*cdms: Cdm) -> ConjunctionEvent:
    return ConjunctionEvent.from_cdms(cdms, event_id="EVENT-7")


def leo_event(hours: float = 25.0) -> ConjunctionEvent:
    return event_of(as_cdm(crossing_conjunction(), hours=hours))


def with_drag(cdm: Cdm, term: float = 0.02) -> Cdm:
    """The CDM, whose primary gives ``CD_AREA_OVER_MASS``."""
    first, second = cdm.object_sections
    extended = Section(
        "segment", (*first.fields, Field("CD_AREA_OVER_MASS", float(term)))
    )
    return Cdm(cdm.header, cdm.relative, (extended, second))


def cell(plan: ManeuverPlan, option: ManeuverOption) -> tuple[int, int]:
    row = int(np.flatnonzero(plan.burn_leads == option.burn_lead)[0])
    column = int(np.flatnonzero(plan.delta_v == option.delta_v)[0])
    return row, column


@pytest.fixture(scope="module", name="plan")
def small_plan() -> ManeuverPlan:
    return plan_maneuver(leo_event(), grid=SMALL_GRID)


@dataclass(frozen=True, eq=False)
class Scripted(PcComputation):
    """Foster's Pc, except on the calls in ``refused``: 1-based, the baseline's
    being the first."""

    refused: frozenset[int] | None = None
    raising: frozenset[int] = frozenset()
    seen: list[Conjunction] = field(default_factory=list)

    @override
    def __call__(self, conjunction: Conjunction) -> PcResult:
        self.seen.append(conjunction)
        count = len(self.seen)
        if count in self.raising:
            raise PcConvergenceError("no convergence")
        result = compute_pc(conjunction, "foster")
        if self.refused is None:
            return result if count == 1 else dataclasses.replace(result, value=0.5)
        return (
            dataclasses.replace(result, value=0.5) if count in self.refused else result
        )


def test_baseline_reproduces_the_cdm_pc(plan: ManeuverPlan) -> None:
    expected = compute_pc(plan.conjunction)
    assert plan.baseline.pc == approx(expected.value, rel=1e-6)
    # Independently: both covariances are isotropic, so the miss is a circular
    # Gaussian of 2 sigma^2 per axis on the encounter plane, and the Pc over the
    # disc of radius R is a noncentral chi-squared probability.
    spread = 2.0 * SIGMA**2
    radius = plan.conjunction.combined_hard_body_radius
    exact = ncx2.cdf(radius**2 / spread, df=2, nc=MISS**2 / spread)
    assert plan.baseline.pc == approx(exact, rel=1e-4)
    assert plan.baseline.delta_v == 0.0
    assert plan.baseline.method == expected.method
    assert plan.baseline.range_at_tca == pytest.approx(MISS, rel=1e-9)
    assert not plan.baseline.meets_target


def test_baseline_applies_no_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    # The CDM's own conjunction: closest_approach_shift has no say in it.
    def no_minimum(r: FloatArray, *_: object, **__: object) -> FloatArray:
        return np.full(r.shape[:-1], np.nan)

    monkeypatch.setattr(planner, "closest_approach_shift", no_minimum)
    plan = plan_maneuver(leo_event(), grid=SMALL_GRID)
    assert plan.baseline.pc == approx(compute_pc(plan.conjunction).value, rel=1e-6)


def test_the_plan_reads_the_latest_cdm() -> None:
    older = as_cdm(crossing_conjunction(), hours=40.0)
    plan = plan_maneuver(
        event_of(older, as_cdm(crossing_conjunction())), grid=SMALL_GRID
    )
    assert plan.burn_leads[0] <= 25 * HOUR


def test_grid_shape_and_axes() -> None:
    grid = dataclasses.replace(DEFAULT_BURN_GRID, steps_per_orbit=1)
    plan = plan_maneuver(leo_event(), grid=grid)
    assert plan.delta_v.shape == (62,)
    assert np.all(np.diff(plan.delta_v) > 0.0)
    assert not np.any(plan.delta_v == 0.0)
    assert np.sum(plan.delta_v > 0.0) == 31
    assert plan.delta_v[31] == pytest.approx(1e-3)
    assert plan.delta_v[-1] == pytest.approx(1.0)
    assert plan.delta_v[:31] == pytest.approx(-plan.delta_v[:30:-1])
    assert np.all(np.diff(plan.burn_leads) < 0.0)
    assert plan.burn_leads[-1] >= MIN_BURN_LEAD
    assert plan.pc.shape == (len(plan.burn_leads), 62)
    assert plan.range_at_tca.shape == plan.pc.shape


def test_a_context_is_the_plan_without_its_grid(plan: ManeuverPlan) -> None:
    alone = prepare_maneuver(leo_event())
    assert not isinstance(alone, ManeuverPlan)
    assert alone.baseline == plan.baseline
    assert (alone.confirmation.method, dict(alone.confirmation.options)) == (
        plan.confirmation.method,
        dict(plan.confirmation.options),
    )
    assert plan.notes[: len(alone.notes)] == alone.notes
    assert not any("cells" in note for note in alone.notes)
    option = plan.prograde
    assert option is not None
    evaluated = evaluate_burn(alone, burn_lead=option.burn_lead, delta_v=option.delta_v)
    assert evaluated == evaluate_burn(
        plan, burn_lead=option.burn_lead, delta_v=option.delta_v
    )


def test_a_context_accepts_a_cdm_inside_the_burn_window() -> None:
    late = leo_event(hours=1.0)
    with pytest.raises(ManeuverError, match="after the burn window closed"):
        plan_maneuver(late)
    assert prepare_maneuver(late).event is late


def test_options_are_named_by_direction(plan: ManeuverPlan) -> None:
    assert plan.prograde is not None
    assert plan.retrograde is not None
    assert plan.options == (plan.prograde, plan.retrograde)
    for direction in BurnDirection:
        option = plan.option(direction)
        assert option is not None
        assert option.direction is direction
        assert math.copysign(1.0, option.delta_v) == direction.sign
    with pytest.raises(ValueError, match="no burn"):
        _ = plan.baseline.direction


def test_burn_window_stops_at_the_cdm_creation(plan: ManeuverPlan) -> None:
    assert plan.burn_leads[0] == pytest.approx(25 * HOUR, abs=1e-3)


def test_burn_window_starts_at_the_decision_point() -> None:
    plan = plan_maneuver(leo_event(100.0), grid=SMALL_GRID)
    assert plan.burn_leads[0] == DECISION_POINT


def test_step_is_an_eighth_of_the_period() -> None:
    grid = BurnGrid(earliest_lead=8 * HOUR, delta_v_per_direction=2)
    plan = plan_maneuver(leo_event(), grid=grid)
    period = orbital_period(plan.conjunction.primary.state)
    assert period is not None
    assert len(plan.burn_leads) > 2
    assert np.diff(plan.burn_leads) == pytest.approx(-period / 8.0)


def test_cdm_too_close_to_tca_is_an_error() -> None:
    with pytest.raises(
        ManeuverError, match=r"EVENT-7.*5\.0 h before TCA.*6\.0 h before TCA"
    ):
        plan_maneuver(leo_event(5.0))


def test_a_cdm_at_the_last_burn_time_has_one_row() -> None:
    plan = plan_maneuver(leo_event(6.0), grid=SMALL_GRID)
    assert plan.burn_leads == pytest.approx([6 * HOUR], abs=1e-3)


def test_minimum_option_per_direction(plan: ManeuverPlan) -> None:
    for option, sign in ((plan.prograde, 1.0), (plan.retrograde, -1.0)):
        assert option is not None
        assert option.delta_v * sign > 0.0
        assert option.meets_target
        assert option.pc < plan.target_pc == TARGET_PC
        row, column = cell(plan, option)
        assert option.range_at_tca == plan.range_at_tca[row, column]
        same_way = plan.delta_v * sign > 0.0
        smaller = np.abs(plan.delta_v) < abs(option.delta_v)
        assert smaller[same_way].any(), "the smallest Δv proves nothing"
        assert np.all(plan.pc[:, same_way & smaller] >= plan.target_pc)


def test_ties_prefer_the_later_burn(plan: ManeuverPlan) -> None:
    assert plan.prograde is not None
    row, column = cell(plan, plan.prograde)
    safe = np.flatnonzero(plan.pc[:, column] < plan.target_pc)
    assert len(safe) > 1, "no tie to break"
    assert row == safe.max()


def test_no_cell_meets_target_gives_none() -> None:
    # Foster's Pc is exactly 0 beyond 12 sigma, and 0 is below any target: only
    # covariances wider than the whole grid leave no such cell.
    event = event_of(as_cdm(crossing_conjunction(1000 * KM)))
    plan = plan_maneuver(event, target_pc=1e-300, grid=SMALL_GRID, compute=FOSTER)
    assert np.all(plan.pc > 1e-300)
    assert plan.prograde is None
    assert plan.retrograde is None


def test_option_confirmed_by_event_method(plan: ManeuverPlan) -> None:
    # Calls: 1 is the baseline, 2 the first prograde candidate.
    compute = Scripted("foster", refused=frozenset({2}))
    confirmed = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=compute)
    assert plan.prograde is not None
    assert confirmed.prograde is not None
    row, column = cell(plan, plan.prograde)
    later = plan.pc[:row, column] < plan.target_pc
    assert later.any(), "the second candidate is not in the same column"
    assert confirmed.prograde.delta_v == plan.prograde.delta_v
    assert confirmed.prograde.burn_lead > plan.prograde.burn_lead
    assert confirmed.retrograde is not None
    assert plan.retrograde is not None
    assert confirmed.retrograde.burn_lead == plan.retrograde.burn_lead
    assert (
        sum("refused" in note and "prograde" in note for note in confirmed.notes) == 1
    )


def test_confirmation_keeps_the_grid_cell_conjunction(plan: ManeuverPlan) -> None:
    compute = Scripted("foster", refused=frozenset())
    plan_maneuver(leo_event(), grid=SMALL_GRID, compute=compute)
    assert plan.prograde is not None
    assert compute.seen[1].miss_distance == pytest.approx(
        plan.prograde.range_at_tca, rel=1e-9
    )
    assert compute.seen[1].tca != compute.seen[0].tca


def test_confirmations_are_bounded() -> None:
    compute = Scripted("foster")
    plan = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=compute)
    assert plan.prograde is None
    assert plan.retrograde is None
    # MAX_CONFIRMATIONS for each direction, after the baseline.
    assert len(compute.seen) == 1 + 2 * MAX_CONFIRMATIONS
    assert MAX_CONFIRMATIONS == 8
    for sign, name in ((1.0, "prograde"), (-1.0, "retrograde")):
        below = np.count_nonzero(plan.pc[:, plan.delta_v * sign > 0.0] < plan.target_pc)
        assert below > MAX_CONFIRMATIONS
        expected = f"{below - MAX_CONFIRMATIONS} candidates left untested"
        assert sum(name in n and expected in n for n in plan.notes) == 1


def test_one_candidate_left_untested_is_singular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=Scripted("foster"))
    below = int(np.count_nonzero(full.pc[:, full.delta_v > 0.0] < full.target_pc))
    module = importlib.import_module("keepout.maneuver.planner")
    monkeypatch.setattr(module, "MAX_CONFIRMATIONS", below - 1)
    plan = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=Scripted("foster"))
    (note,) = (n for n in plan.notes if n.startswith("prograde: of "))
    assert f"of {below} cells below the target" in note
    assert note.endswith(f"after {below - 1} tries with 1 candidate left untested")


def test_a_search_that_ends_early_does_not_claim_untested_candidates() -> None:
    compute = Scripted("foster", refused=frozenset({2}))
    plan = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=compute)
    assert plan.prograde is not None
    assert not any("untested" in note for note in plan.notes)


def test_a_failed_confirmation_moves_to_the_next_candidate(plan: ManeuverPlan) -> None:
    compute = Scripted("foster", refused=frozenset(), raising=frozenset({2}))
    confirmed = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=compute)
    assert plan.prograde is not None
    assert confirmed.prograde is not None
    assert confirmed.prograde.burn_lead > plan.prograde.burn_lead
    assert any("prograde" in note and "failed" in note for note in confirmed.notes)


def test_conservative_basis_decides_on_the_upper_end() -> None:
    def with_interval(conjunction: Conjunction) -> PcResult:
        result = compute_pc(conjunction, "foster")
        return dataclasses.replace(
            result,
            method="montecarlo",
            interval=(0.0, result.value + 2.0 * TARGET_PC),
        )

    @dataclass(frozen=True, eq=False)
    class Wide(PcComputation):
        @override
        def __call__(self, conjunction: Conjunction) -> PcResult:
            return with_interval(conjunction)

        @override
        def resolved(self, result: PcResult) -> PcComputation:
            return self

    plan = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=Wide("foster"))
    assert plan.baseline.pc > 2.0 * TARGET_PC
    assert plan.prograde is None
    loose = plan_maneuver(
        leo_event(),
        grid=SMALL_GRID,
        compute=Wide("foster"),
        thresholds=Thresholds(basis="estimate"),
    )
    assert loose.prograde is not None
    assert loose.prograde.method == "montecarlo"


@pytest.mark.slow
def test_auto_fallback_keeps_its_seed_and_cap_in_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_convergence(*_: object, **__: object) -> PcResult:
        raise PcConvergenceError("no convergence")

    # Real, seeded Monte Carlo, with a cap that keeps the runs to a test's length.
    cap = 20_000
    auto_hall_integration(monkeypatch, no_convergence)
    auto_monte_carlo_budget(monkeypatch, cap)
    runs = recorded_runs(monkeypatch)
    event = event_of(as_cdm(slow_conjunction()))
    first = plan_maneuver(event, target_pc=1e-3, grid=SMALL_GRID)
    assert first.baseline.method == "montecarlo"
    expected = {"max_samples": cap, "seed": AUTO_MC_SEED}
    assert first.confirmation.options == expected
    confirmed = [options for method, options in runs if method == "montecarlo"]
    assert confirmed, "no confirmation ran"
    assert all(options == expected for options in confirmed)
    assert first.prograde is not None
    assert first.prograde.method == "montecarlo"

    runs.clear()
    evaluated = evaluate_burn(
        first, burn_lead=first.prograde.burn_lead, delta_v=first.prograde.delta_v
    )
    assert runs == [("montecarlo", expected)]
    # One burn integrates apart from the grid, by under a micrometre here: the
    # same seed then gives the same Pc, up to a sample crossing the radius.
    assert evaluated.pc == pytest.approx(first.prograde.pc, rel=0.05)


def test_explicit_options_are_what_confirms_and_evaluates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True, eq=False)
    class Counted(PcComputation):
        calls: list[Conjunction] = field(default_factory=list)

        @override
        def __call__(self, conjunction: Conjunction) -> PcResult:
            self.calls.append(conjunction)
            return super().__call__(conjunction)

    runs = recorded_runs(monkeypatch)
    compute = Counted("foster", {"rtol": 1e-6})
    plan = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=compute)
    assert plan.confirmation is compute
    assert plan.prograde is not None
    # The baseline and a confirmation per direction at least; the grid's own
    # Foster runs take the defaults.
    expected: tuple[str, dict[str, object]] = ("foster", {"rtol": 1e-6})
    assert len(compute.calls) >= 3
    assert runs.count(expected) == len(compute.calls)
    runs.clear()
    evaluate_burn(
        plan, burn_lead=plan.prograde.burn_lead, delta_v=plan.prograde.delta_v
    )
    assert runs == [expected]


def test_a_hall3d_failure_on_a_candidate_falls_back_to_monte_carlo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = 20_000
    auto_monte_carlo_budget(monkeypatch, cap)
    spy = hall3d_failing(monkeypatch)
    event = event_of(as_cdm(slow_conjunction()))
    plan = plan_maneuver(event, target_pc=1e-3, grid=SMALL_GRID)
    assert plan.baseline.method == "hall3d"
    fallback = plan.confirmation.fallback
    assert fallback is not None
    assert fallback.method == "montecarlo"
    assert fallback.options == {"max_samples": cap, "seed": AUTO_MC_SEED}
    option = plan.option(BurnDirection.PROGRADE)
    assert option is not None
    assert option.method == "montecarlo"
    expected = "hall3d failed (Nc is not finite); montecarlo used instead"
    assert expected in option.notes
    assert f"prograde: {expected}" in plan.notes
    assert not any("failed on" in note for note in plan.notes)

    spy.armed = True
    evaluated = evaluate_burn(plan, burn_lead=option.burn_lead, delta_v=option.delta_v)
    assert evaluated.method == "montecarlo"
    assert expected in evaluated.notes


def test_an_explicit_hall3d_gets_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # The baseline is the first hall3d run; the first prograde candidate fails.
    spy = hall3d_failing(monkeypatch, skip=1)
    event = event_of(as_cdm(slow_conjunction()))
    plan = plan_maneuver(event, grid=SMALL_GRID, compute=computation("hall3d"))
    assert plan.confirmation.fallback is None
    assert "montecarlo" not in spy.runs
    assert plan.prograde is not None
    assert plan.prograde.method == "hall3d"
    assert not any("used instead" in note for note in plan.notes)
    assert any(
        note.startswith("prograde: ") and "failed on 1" in note for note in plan.notes
    )

    spy.armed = True
    with pytest.raises(PcConvergenceError, match="Nc is not finite"):
        evaluate_burn(
            plan, burn_lead=plan.prograde.burn_lead, delta_v=plan.prograde.delta_v
        )


def test_a_foster_baseline_gets_no_fallback(plan: ManeuverPlan) -> None:
    assert plan.baseline.method == "foster"
    assert plan.confirmation.fallback is None


def test_evaluate_burn_returns_the_burns_own_diagnostics() -> None:
    @dataclass(frozen=True, eq=False)
    class Noting(PcComputation):
        @override
        def __call__(self, conjunction: Conjunction) -> PcResult:
            result = compute_pc(conjunction, "foster")
            miss = conjunction.miss_distance
            note = () if miss == pytest.approx(MISS) else (f"burnt {miss:.0f}",)
            return dataclasses.replace(result, diagnostics=("shared", *note))

    plan = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=Noting("foster"))
    assert "shared" in plan.notes
    assert plan.baseline.notes == ()
    option = plan.prograde
    assert option is not None
    assert option.notes == (f"burnt {option.range_at_tca:.0f}",)
    evaluated = evaluate_burn(plan, burn_lead=option.burn_lead, delta_v=option.delta_v)
    assert evaluated.notes == (f"burnt {evaluated.range_at_tca:.0f}",)


def test_assumed_covariance_is_carried() -> None:
    cdm = as_cdm(crossing_conjunction(), method=CovarianceMethod.DEFAULT)
    plan = plan_maneuver(event_of(cdm), grid=SMALL_GRID)
    assert plan.assumed_covariance
    assert any("assumed" in note for note in plan.notes)
    calculated = plan_maneuver(leo_event(), grid=SMALL_GRID)
    assert not calculated.assumed_covariance
    assert not any("assumed" in note for note in calculated.notes)


def test_position_only_covariance_stays_on_foster() -> None:
    # Slow, so that "auto" would go to Hall's 3-D method were velocities known.
    cdm = as_cdm(slow_conjunction(velocity_sigma=0.0))
    plan = plan_maneuver(event_of(cdm), grid=SMALL_GRID)
    assert plan.baseline.method == "foster"
    assert any("only in position" in note for note in plan.notes)


def test_unknown_covariance_is_an_error() -> None:
    conjunction = crossing_conjunction()
    secondary = dataclasses.replace(conjunction.secondary, covariance=None)
    unknown = Conjunction(conjunction.primary, secondary, 20.0)
    with pytest.raises(ManeuverError, match=r"EVENT-7.*secondary \(2002\)"):
        plan_maneuver(event_of(as_cdm(unknown)), grid=SMALL_GRID)


def test_a_missing_hard_body_radius_is_said_as_elsewhere() -> None:
    conjunction = crossing_conjunction()
    primary = dataclasses.replace(conjunction.primary, hard_body_radius=None)
    secondary = dataclasses.replace(conjunction.secondary, hard_body_radius=None)
    event = event_of(as_cdm(Conjunction(primary, secondary, 20.0)))
    with pytest.raises(HardBodyRadiusError):
        plan_maneuver(event, grid=SMALL_GRID)
    assert plan_maneuver(
        event, grid=SMALL_GRID, combined_hard_body_radius=20.0
    ).prograde


def test_primary_that_cannot_manoeuvre_is_an_error() -> None:
    cdm = as_cdm(crossing_conjunction(), maneuverable=Maneuverable.NO)
    with pytest.raises(ManeuverError, match=r"EVENT-7.*1001.*MANEUVERABLE = NO"):
        plan_maneuver(event_of(cdm), grid=SMALL_GRID)
    unknown = as_cdm(crossing_conjunction(), maneuverable=Maneuverable.NOT_APPLICABLE)
    assert plan_maneuver(event_of(unknown), grid=SMALL_GRID).prograde is not None


def test_primary_on_an_unbound_orbit_is_an_error() -> None:
    conjunction = crossing_conjunction()
    state = StateVector(
        TCA, conjunction.primary.state.r, 2.0 * conjunction.primary.state.v, "GCRF"
    )
    fast = dataclasses.replace(conjunction.primary, state=state)
    event = event_of(as_cdm(Conjunction(fast, conjunction.secondary, 20.0)))
    with pytest.raises(ManeuverError, match=r"EVENT-7.*no orbital period"):
        plan_maneuver(event, grid=SMALL_GRID)


def test_drag_used_only_with_cd_area_over_mass(
    plan: ManeuverPlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    terms: list[float | None] = []

    def spy(
        state: StateVector,
        burn_leads: npt.ArrayLike,
        delta_v: npt.ArrayLike,
        force: ForceModel,
    ) -> BurnEffects:
        terms.append(force.drag_term)
        return burn_effects(state, burn_leads, delta_v, force)

    monkeypatch.setattr(planner, "burn_effects", spy)
    without = plan_maneuver(leo_event(), grid=SMALL_GRID)
    assert sum("drag is off" in note for note in without.notes) == 1
    event = event_of(with_drag(as_cdm(crossing_conjunction())))
    dragged = plan_maneuver(event, grid=SMALL_GRID)
    assert not any("drag is off" in note for note in dragged.notes)
    event = event_of(with_drag(as_cdm(crossing_conjunction()), 0.0))
    zero = plan_maneuver(event, grid=SMALL_GRID)
    assert sum("drag is off" in note for note in zero.notes) == 1
    assert terms == [None, 0.02, None]


def drag_note(cdm: Cdm) -> str:
    plan = plan_maneuver(event_of(cdm), grid=SMALL_GRID)
    (note,) = [note for note in plan.notes if "drag is off" in note]
    return note


@pytest.mark.parametrize("term", [None, 0.0])
def test_drag_off_warns_below_450_km(term: float | None) -> None:
    cdm = as_cdm(crossing_conjunction(radius=EARTH_RADIUS + 400 * KM))
    note = drag_note(cdm if term is None else with_drag(cdm, term))
    assert "perigee at 400 km" in note
    assert "0.18" in note
    assert "1.29" in note
    assert "bound" in note


def test_drag_off_does_not_warn_above_450_km() -> None:
    note = drag_note(as_cdm(crossing_conjunction(radius=EARTH_RADIUS + 500 * KM)))
    assert "perigee" not in note


def test_the_drag_warning_goes_by_perigee_not_by_altitude_at_tca() -> None:
    conjunction = crossing_conjunction(radius=EARTH_RADIUS + 500 * KM)
    apogee = EARTH_RADIUS + 500 * KM
    axis = apogee - 50 * KM
    speed = math.sqrt(EARTH_MU * (2.0 / apogee - 1.0 / axis))
    state = StateVector(
        TCA, conjunction.primary.state.r, speed * direction(INCLINATION), "GCRF"
    )
    primary = dataclasses.replace(conjunction.primary, state=state)
    cdm = as_cdm(Conjunction(primary, conjunction.secondary, 20.0))
    assert "perigee at 400 km" in drag_note(cdm)


def test_drag_on_gives_no_warning_at_any_altitude() -> None:
    cdm = as_cdm(crossing_conjunction(radius=EARTH_RADIUS + 400 * KM))
    plan = plan_maneuver(event_of(with_drag(cdm)), grid=SMALL_GRID)
    assert not any("perigee" in note for note in plan.notes)


def test_a_negative_drag_term_is_an_error() -> None:
    event = event_of(with_drag(as_cdm(crossing_conjunction()), -0.02))
    with pytest.raises(ManeuverError, match=r"EVENT-7.*CD_AREA_OVER_MASS"):
        plan_maneuver(event, grid=SMALL_GRID)


@pytest.mark.slow
def test_drag_barely_changes_the_grid(
    record_property: Callable[[str, object], None],
) -> None:
    # 500 km, where the claim is tight: 0.03 here and 0.003 at 630 km. It does
    # not hold lower down: 0.18 at 400 km and 1.3 at 300 km (Pc near 1e-28).
    cdm = as_cdm(crossing_conjunction(radius=EARTH_RADIUS + 500 * KM))
    plain = plan_maneuver(event_of(cdm), compute=FOSTER)
    dragged = event_of(with_drag(cdm))
    drag = plan_maneuver(dragged, compute=FOSTER)
    difference = np.abs(
        np.log10(np.maximum(plain.pc, PC_FLOOR))
        - np.log10(np.maximum(drag.pc, PC_FLOOR))
    )
    largest = float(np.max(difference))
    record_property("max_log10_pc_difference", largest)
    assert largest < 0.1


def test_a_burn_into_the_earth_names_the_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def falling(*_: object) -> None:
        raise PropagationError("an orbit fell below the Earth's surface")

    monkeypatch.setattr(planner, "burn_effects", falling)
    with pytest.raises(ManeuverError, match=r"EVENT-7.*below the Earth's surface"):
        plan_maneuver(leo_event(), grid=SMALL_GRID)


INTO_THE_SURFACE: Final = BurnGrid(
    steps_per_orbit=1, delta_v_per_direction=6, largest_delta_v=500.0
)
""":data:`SMALL_GRID` up to 500 m/s, which retrograde takes the primary 1900 km
below the surface; 36 m/s next to it takes 130 km off its perigee."""


def test_plan_counts_failed_cells_in_its_notes() -> None:
    plan = plan_maneuver(leo_event(), grid=INTO_THE_SURFACE, compute=FOSTER)
    rows = len(plan.burn_leads)
    assert np.isnan(plan.pc[:, 0]).all()
    assert np.isnan(plan.range_at_tca[:, 0]).all()
    assert np.isfinite(plan.pc[:, 1:]).any()
    no_pc = int(np.count_nonzero(np.isnan(plan.pc)))
    note = (
        f"{no_pc} of {plan.pc.size} cells have no Pc and are never options: "
        f"{no_pc - rows} without a closest approach to trust, "
        f"{rows} where the burn or Foster's Pc failed"
    )
    assert note in plan.notes
    assert plan.retrograde is None or plan.retrograde.delta_v != -500.0


def test_evaluate_burn_refuses_a_burn_into_the_surface(plan: ManeuverPlan) -> None:
    with pytest.raises(
        ManeuverError, match=r"EVENT-7.*-500 m/s.*7\.0 h before TCA.*surface"
    ):
        evaluate_burn(plan, burn_lead=7 * HOUR, delta_v=-500.0)


def test_evaluate_burn_matches_the_grid_cell(plan: ManeuverPlan) -> None:
    option = plan.prograde
    assert option is not None
    row, column = cell(plan, option)
    evaluated = evaluate_burn(plan, burn_lead=option.burn_lead, delta_v=option.delta_v)
    assert evaluated.burn_lead == option.burn_lead
    assert evaluated.delta_v == option.delta_v
    assert evaluated.method == plan.baseline.method
    assert evaluated.meets_target
    assert evaluated.pc == approx(float(plan.pc[row, column]), rel=1e-3)
    assert evaluated.range_at_tca == pytest.approx(
        plan.range_at_tca[row, column], rel=1e-3
    )


def test_evaluate_burn_keeps_the_force_of_the_plan() -> None:
    event = event_of(with_drag(as_cdm(crossing_conjunction())))
    plan = plan_maneuver(event, grid=SMALL_GRID)
    option = plan.prograde
    assert option is not None
    evaluated = evaluate_burn(plan, burn_lead=option.burn_lead, delta_v=option.delta_v)
    assert evaluated.pc == approx(option.pc, rel=1e-3)


def test_evaluate_burn_says_when_a_burn_misses_the_target(plan: ManeuverPlan) -> None:
    missed = evaluate_burn(plan, burn_lead=7 * HOUR, delta_v=1e-3)
    assert not missed.meets_target
    assert missed.pc >= plan.target_pc


def test_evaluate_burn_uses_the_thresholds_basis(plan: ManeuverPlan) -> None:
    @dataclass(frozen=True, eq=False)
    class Wide(PcComputation):
        @override
        def __call__(self, conjunction: Conjunction) -> PcResult:
            result = compute_pc(conjunction, "foster")
            return dataclasses.replace(result, interval=(0.0, 0.25))

    wide = dataclasses.replace(plan, confirmation=Wide("foster"))
    assert evaluate_burn(wide, burn_lead=7 * HOUR, delta_v=1e-3).pc == 0.25
    estimate = Thresholds(basis="estimate")
    on_estimate = dataclasses.replace(wide, thresholds=estimate)
    pc = evaluate_burn(on_estimate, burn_lead=7 * HOUR, delta_v=1e-3).pc
    # Wide's estimate is Foster's value, untouched.
    foster = dataclasses.replace(plan, confirmation=FOSTER)
    assert pc == evaluate_burn(foster, burn_lead=7 * HOUR, delta_v=1e-3).pc


def test_evaluate_burn_rejects_a_burn_before_the_cdm(plan: ManeuverPlan) -> None:
    with pytest.raises(ManeuverError, match=r"EVENT-7.*26\.0 h before TCA"):
        evaluate_burn(plan, burn_lead=26 * HOUR, delta_v=0.1)


def test_evaluate_burn_refuses_a_cell_without_a_minimum() -> None:
    plan = plan_maneuver(
        event_of(as_cdm(slow_conjunction())), grid=SMALL_GRID, compute=FOSTER
    )
    row, column = np.argwhere(np.isnan(plan.pc))[0]
    with pytest.raises(ManeuverError, match=r"EVENT-7.*closest approach"):
        evaluate_burn(
            plan,
            burn_lead=float(plan.burn_leads[row]),
            delta_v=float(plan.delta_v[column]),
        )


@pytest.mark.parametrize("target", [0.0, 1.0, -1e-5, 1.5, math.nan, math.inf])
def test_target_outside_zero_one_is_refused(target: float) -> None:
    with pytest.raises(ValueError, match="target_pc must be in"):
        plan_maneuver(leo_event(), target_pc=target, grid=SMALL_GRID)


def test_slow_encounter_cells_are_nan_and_noted() -> None:
    plan = plan_maneuver(
        event_of(as_cdm(slow_conjunction())), grid=SMALL_GRID, compute=FOSTER
    )
    empty = int(np.count_nonzero(np.isnan(plan.pc)))
    assert 0 < empty <= plan.pc.size
    assert np.array_equal(np.isnan(plan.pc), np.isnan(plan.range_at_tca))
    counting = [note for note in plan.notes if f"{empty} of {plan.pc.size}" in note]
    assert len(counting) == 1
    (slow,) = (note for note in plan.notes if "only a re-screen shows" in note)
    assert "within 1/20 of an orbit of TCA" in slow
    assert "hours" in slow
    assert not any("tens of metres" in note for note in plan.notes)
    assert not any("0.2 m" in note for note in plan.notes)
    assert plan.options
    for option in plan.options:
        assert not math.isnan(plan.pc[cell(plan, option)])


def test_a_slow_encounter_is_confirmed_by_its_own_method() -> None:
    plan = plan_maneuver(event_of(as_cdm(slow_conjunction())), grid=SMALL_GRID)
    assert plan.baseline.method == "hall3d"
    assert plan.prograde is not None
    assert plan.prograde.method == "hall3d"
    assert plan.prograde.meets_target


def test_fast_encounter_has_no_slow_note_and_no_empty_cells(plan: ManeuverPlan) -> None:
    assert not np.isnan(plan.pc).any()
    assert not any("only a re-screen shows" in note for note in plan.notes)
    assert not any(" cells " in note for note in plan.notes)


def test_a_cell_without_a_minimum_is_never_an_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_minimum(r: FloatArray, *_: object, **__: object) -> FloatArray:
        return np.full(r.shape[:-1], np.nan)

    monkeypatch.setattr(planner, "closest_approach_shift", no_minimum)
    plan = plan_maneuver(leo_event(), grid=SMALL_GRID)
    assert np.isnan(plan.pc).all()
    assert plan.prograde is None
    assert plan.retrograde is None


def test_a_cell_whose_pc_fails_is_nan_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Conjunction] = []

    def flaky(conjunction: Conjunction) -> PcResult:
        calls.append(conjunction)
        if len(calls) <= 3:
            raise PcConvergenceError("no convergence")
        return FOSTER(conjunction)

    @dataclass(frozen=True, eq=False)
    class Flaky(PcComputation):
        @override
        def __call__(self, conjunction: Conjunction) -> PcResult:
            return flaky(conjunction)

    monkeypatch.setattr(planner, "_CELL_PC", Flaky("foster"))
    plan = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=FOSTER)
    assert np.count_nonzero(np.isnan(plan.pc)) == 3
    assert np.count_nonzero(np.isnan(plan.range_at_tca)) == 0
    assert sum(f"3 of {plan.pc.size}" in note for note in plan.notes) == 1


def test_the_notes_repeat_what_the_baseline_says() -> None:
    cdm = as_cdm(slow_conjunction(velocity_sigma=0.0))
    plan = plan_maneuver(event_of(cdm), grid=SMALL_GRID)
    diagnostics = compute_pc(plan.conjunction).diagnostics
    assert diagnostics
    assert set(diagnostics) <= set(plan.notes)


def test_a_synthetic_cdm_is_planned_and_says_what_it_rests_on() -> None:
    # As the synthetic operator writes it: assumed and position-only.
    conjunction = slow_conjunction(velocity_sigma=0.0)
    cdm = as_cdm(conjunction, method=CovarianceMethod.DEFAULT)
    plan = plan_maneuver(event_of(cdm), grid=SMALL_GRID)
    assert plan.conjunction.position_only_covariance == ("primary", "secondary")
    assert plan.assumed_covariance
    assert plan.baseline.method == "foster"
    assert sum("assumed covariance" in note for note in plan.notes) == 1
    assert any("only in position" in note for note in plan.notes)
    option = plan.prograde
    assert option is not None
    assert option.method == "foster"
    evaluated = evaluate_burn(plan, burn_lead=option.burn_lead, delta_v=option.delta_v)
    assert evaluated.pc == approx(option.pc, rel=1e-3)


def test_a_chosen_option_adds_its_own_diagnostics() -> None:
    @dataclass(frozen=True, eq=False)
    class Noting(PcComputation):
        @override
        def __call__(self, conjunction: Conjunction) -> PcResult:
            result = compute_pc(conjunction, "foster")
            miss = conjunction.miss_distance
            note = () if miss == pytest.approx(MISS) else (f"burnt {miss:.0f}",)
            return dataclasses.replace(result, diagnostics=("shared", *note))

    plan = plan_maneuver(leo_event(), grid=SMALL_GRID, compute=Noting("foster"))
    assert plan.prograde is not None
    assert plan.retrograde is not None
    assert f"prograde: burnt {plan.prograde.range_at_tca:.0f}" in plan.notes
    assert f"retrograde: burnt {plan.retrograde.range_at_tca:.0f}" in plan.notes
    assert plan.notes.count("shared") == 1
    assert not any(note.endswith(": shared") for note in plan.notes)


def test_the_arrays_are_read_only(plan: ManeuverPlan) -> None:
    for array in (plan.burn_leads, plan.delta_v, plan.pc, plan.range_at_tca):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.0


def test_the_arrays_must_fit_the_axes(plan: ManeuverPlan) -> None:
    with pytest.raises(ValueError, match="pc and range_at_tca must have shape"):
        dataclasses.replace(plan, pc=plan.pc[:-1])


class GridOptions(TypedDict, total=False):
    """Fields of a :class:`BurnGrid`."""

    earliest_lead: float
    latest_lead: float
    steps_per_orbit: int
    smallest_delta_v: float
    largest_delta_v: float
    delta_v_per_direction: int


@pytest.mark.parametrize(
    ("options", "name"),
    [
        ({"earliest_lead": 0.0}, "earliest_lead"),
        ({"latest_lead": -1.0}, "latest_lead"),
        ({"earliest_lead": HOUR, "latest_lead": 2 * HOUR}, "earliest_lead"),
        ({"steps_per_orbit": 0}, "steps_per_orbit"),
        ({"smallest_delta_v": 0.0}, "smallest_delta_v"),
        ({"largest_delta_v": math.inf}, "largest_delta_v"),
        ({"smallest_delta_v": 2.0}, "largest_delta_v"),
        ({"delta_v_per_direction": 0}, "delta_v_per_direction"),
    ],
)
def test_a_grid_must_make_sense(options: GridOptions, name: str) -> None:
    with pytest.raises(ValueError, match=name):
        BurnGrid(**options)


def test_the_default_grid_is_the_documented_one() -> None:
    documented = BurnGrid(
        earliest_lead=DECISION_POINT,
        latest_lead=6 * HOUR,
        steps_per_orbit=8,
        smallest_delta_v=1e-3,
        largest_delta_v=1.0,
        delta_v_per_direction=31,
    )
    assert documented == DEFAULT_BURN_GRID
    assert TARGET_PC == 1e-5
