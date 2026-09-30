import importlib
import math

import numpy as np
import pytest
from builders import EPOCH, LEO_R, LEO_V, head_on, known, object_state, shifted_back

from keepout._types import StateVector
from keepout.conjunction import Conjunction
from keepout.covariance import make_covariance
from keepout.errors import CovarianceError, GeometryError, PcConvergenceError
from keepout.objects import ObjectState
from keepout.orbits import orbital_period
from keepout.pc import PcResult, alfano, compute_pc, elrod, foster, frisbee
from keepout.pc.api import (
    AUTO_MC_SAMPLES,
    AUTO_MC_SEED,
    DURATION_LIMIT,
    SIGMA_EXTENT,
    MethodChoice,
    encounter_duration,
)
from keepout.pc.montecarlo import montecarlo

api = importlib.import_module("keepout.pc.api")

_period = orbital_period(StateVector(EPOCH, LEO_R, LEO_V, "GCRF"))
assert _period is not None
LEO_PERIOD: float = _period


def test_auto_uses_foster() -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    result = compute_pc(conjunction)
    assert result.method == "foster"
    assert result.value == foster(conjunction.plane, 5.0).value


def test_elrod_can_be_chosen() -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    result = compute_pc(conjunction, "elrod")
    assert result.method == "elrod"
    assert result.value == elrod(conjunction.plane, 5.0).value


def test_options_reach_the_method() -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    result = compute_pc(conjunction, "elrod", order=16)
    assert result.details["order"] == 16
    assert result.value == elrod(conjunction.plane, 5.0, order=16).value
    loose = compute_pc(conjunction, "foster", rtol=1e-4)
    assert loose.details["evaluations"] < compute_pc(conjunction).details["evaluations"]


def test_option_of_another_method_is_rejected() -> None:
    with pytest.raises(TypeError, match="order"):
        compute_pc(head_on(30.0, 10.0, 20.0, 5.0), "foster", order=16)


def test_unknown_method_is_rejected() -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    with pytest.raises(ValueError, match="unknown Pc method 'coppola'"):
        compute_pc(conjunction, "coppola")  # type: ignore[arg-type]


def test_fast_leo_encounter_has_no_warnings() -> None:
    result = compute_pc(head_on(30.0, 10.0, 20.0, 5.0))
    assert result.diagnostics == ()
    # head_on: v_rel along GCRF -y where the primary's sigma is 1 m, the
    # secondary's 0: 2 (5 * 1 m + 5 m) / (2 * 7546 m/s).
    assert result.details["encounter_duration_s"] == pytest.approx(
        2 * (SIGMA_EXTENT * 1.0 + 5.0) / (2 * 7546.0), rel=1e-12
    )


def test_covariance_repairs_come_first_labelled_by_object() -> None:
    base = head_on(30.0, 10.0, 20.0, 5.0)
    asymmetric = known(base.primary.covariance).matrix.copy()
    asymmetric[0, 1] += 1.0
    repaired = ObjectState(
        "12345", base.primary.state, make_covariance(asymmetric, "GCRF", EPOCH)
    )
    result = compute_pc(Conjunction(repaired, base.secondary, 5.0))
    assert result.diagnostics == (
        "primary (12345): symmetrised covariance (max asymmetry 1)",
    )


def along_track_conjunction(duration: float) -> Conjunction:
    """Head-on LEO conjunction whose encounter lasts ``duration`` seconds.

    The relative velocity is along GCRF y, so the along-track sigma sets the
    duration: ``2 (SIGMA_EXTENT sigma + R) / v``, with R = 5 m.
    """
    radius, speed = 5.0, 2 * 7546.0
    sigma = (duration * speed / 2.0 - radius) / SIGMA_EXTENT
    primary = object_state(LEO_R, LEO_V, np.diag([100.0, sigma**2, 100.0]))
    secondary = object_state(
        LEO_R + np.array([30.0, 0.0, 0.0]), -LEO_V, np.zeros((3, 3))
    )
    return Conjunction(primary, secondary, radius)


@pytest.mark.parametrize(("fraction", "warns"), [(0.99, False), (1.01, True)])
def test_duration_warning_starts_at_the_limit(fraction: float, warns: bool) -> None:
    conjunction = along_track_conjunction(fraction * DURATION_LIMIT * LEO_PERIOD)
    assert encounter_duration(conjunction) == pytest.approx(
        fraction * DURATION_LIMIT * LEO_PERIOD, rel=1e-9
    )
    diagnostics = compute_pc(conjunction).diagnostics
    assert (
        any(d.startswith("2-D Pc assumptions questionable") for d in diagnostics)
        is warns
    )


def test_unbound_primary_gets_no_duration_warning() -> None:
    # 15 km/s at 7000 km is above escape speed: there is no period to compare.
    fast = np.array([0.0, 15000.0, 0.0])
    primary = object_state(LEO_R, fast, 1e8 * np.eye(3))
    secondary = object_state(
        LEO_R + np.array([30.0, 0.0, 0.0]), fast + np.array([0.0, 0.0, 1.0]), np.eye(3)
    )
    assert compute_pc(Conjunction(primary, secondary, 5.0)).diagnostics == ()


def test_auto_keeps_foster_for_a_questionable_unbound_encounter() -> None:
    # Unbound primary, secondary 10 m off closest approach along the relative
    # velocity (z): a 2-D assumption is questionable, but Hall needs elliptic
    # orbits, so Foster stays, with the note and without a 2-D comparison.
    fast = np.array([0.0, 15000.0, 0.0])
    primary = object_state(LEO_R, fast, 1e8 * np.eye(3))
    secondary = object_state(
        LEO_R + np.array([30.0, 0.0, 10.0]),
        fast + np.array([0.0, 0.0, 1.0]),
        np.eye(3),
    )
    result = compute_pc(Conjunction(primary, secondary, 5.0))
    assert result.method == "foster"
    (message,) = result.diagnostics
    assert message.startswith("states are 10 m from closest approach")
    assert "pc_2d" not in result.details


def test_states_away_from_tca_are_reported() -> None:
    # Secondary 10 m along the relative velocity from closest approach.
    primary = object_state(LEO_R, LEO_V, 100.0 * np.eye(3))
    secondary = object_state(LEO_R + np.array([30.0, -10.0, 0.0]), -LEO_V, np.eye(3))
    (message,) = compute_pc(Conjunction(primary, secondary, 5.0)).diagnostics
    assert message.startswith("states are 10 m from closest approach")
    assert message.endswith("; 2-D Pc uses the straight-line closest approach")


def test_large_sigma_across_a_slow_track_does_not_warn() -> None:
    # 20 m/s along z; 2 km sigma along x but 1 m along z: crossing ~0.5 s.
    primary = object_state(LEO_R, LEO_V, np.diag([4e6, 4e6, 1.0]))
    secondary = object_state(
        LEO_R + np.array([20.0, 0.0, 0.0]),
        LEO_V + np.array([0.0, 0.0, 20.0]),
        np.eye(3),
    )
    result = compute_pc(Conjunction(primary, secondary, 5.0))
    assert result.diagnostics == ()
    assert result.details["encounter_duration_s"] == pytest.approx(
        2 * (SIGMA_EXTENT * math.sqrt(2.0) + 5.0) / 20.0, rel=1e-9
    )


def without_covariance(item: ObjectState) -> ObjectState:
    return ObjectState(item.object_id, item.state, None)


def test_auto_uses_frisbee_when_one_covariance_is_unknown() -> None:
    base = head_on(300.0, 100.0, 200.0, 5.0)
    conjunction = Conjunction(base.primary, without_covariance(base.secondary), 5.0)
    result = compute_pc(conjunction)
    assert result.method == "frisbee"
    assert result.diagnostics[0] == "secondary (SEC): covariance unknown"
    assert "upper bound" in result.diagnostics[1]


def test_auto_uses_alfano_when_both_covariances_are_unknown() -> None:
    base = head_on(300.0, 100.0, 200.0, 5.0)
    conjunction = Conjunction(
        without_covariance(base.primary), without_covariance(base.secondary), 5.0
    )
    result = compute_pc(conjunction)
    assert result.method == "alfano"
    assert result.value == pytest.approx(
        5.0 / 300.0 * math.sqrt(2.0 / (math.pi * math.e)), rel=1e-3
    )


def test_frisbee_can_be_chosen() -> None:
    base = head_on(300.0, 100.0, 200.0, 5.0)
    conjunction = Conjunction(base.primary, without_covariance(base.secondary), 5.0)
    result = compute_pc(conjunction, "frisbee")
    assert result.method == "frisbee"
    assert result.value == frisbee(conjunction.plane, 5.0).value
    assert "encounter_duration_s" not in result.details
    assert result.diagnostics[0] == "secondary (SEC): covariance unknown"
    # A fast encounter at TCA: no applicability note.
    assert result.diagnostics[1].startswith("maximum Pc over the unknown secondary")
    assert len(result.diagnostics) == 2


@pytest.mark.parametrize("method", ["auto", "frisbee"])
def test_maximum_pc_keeps_a_duration_warning(method: MethodChoice) -> None:
    # The duration from the known covariance alone is a lower bound: when it
    # already exceeds the limit, the warning holds.
    slow = along_track_conjunction(1.5 * DURATION_LIMIT * LEO_PERIOD)
    conjunction = Conjunction(slow.primary, without_covariance(slow.secondary), 5.0)
    result = compute_pc(conjunction, method)
    assert result.method == "frisbee"
    assert result.diagnostics[1].startswith(
        "2-D Pc assumptions questionable: the encounter lasts at least about"
    )
    assert "encounter_duration_s" not in result.details


@pytest.mark.parametrize("method", ["auto", "alfano"])
def test_maximum_pc_keeps_the_closest_approach_offset_note(
    method: MethodChoice,
) -> None:
    # The offset does not depend on the covariances.
    base = shifted_back(head_on(300.0, 100.0, 200.0, 5.0), 1.0)
    conjunction = Conjunction(
        without_covariance(base.primary), without_covariance(base.secondary), 5.0
    )
    result = compute_pc(conjunction, method)
    assert result.method == "alfano"
    assert any(
        d.startswith("states are 1.51e+04 m from closest") for d in result.diagnostics
    )
    assert "encounter_duration_s" not in result.details


def test_alfano_can_be_chosen() -> None:
    base = head_on(300.0, 100.0, 200.0, 5.0)
    conjunction = Conjunction(
        without_covariance(base.primary), without_covariance(base.secondary), 5.0
    )
    result = compute_pc(conjunction, "alfano")
    assert result.method == "alfano"
    assert result.value == alfano(conjunction.plane, 5.0).value
    assert "encounter_duration_s" not in result.details


def test_options_reach_a_three_d_method() -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    result = compute_pc(conjunction, "montecarlo", samples=2_000, seed=1)
    assert result.method == "montecarlo"
    assert result.details["samples"] == 2_000
    assert result.value == montecarlo(conjunction, samples=2_000, seed=1).value


@pytest.mark.parametrize(
    "method", ["foster", "elrod", "dilution", "hall", "montecarlo"]
)
def test_methods_needing_both_covariances_refuse_an_unknown_one(
    method: MethodChoice,
) -> None:
    base = head_on(300.0, 100.0, 200.0, 5.0)
    conjunction = Conjunction(base.primary, without_covariance(base.secondary), 5.0)
    with pytest.raises(CovarianceError, match="needs both covariances"):
        compute_pc(conjunction, method)


def test_auto_uses_hall_for_a_slow_encounter_and_keeps_the_2d_pc() -> None:
    conjunction = along_track_conjunction(1.5 * DURATION_LIMIT * LEO_PERIOD)
    result = compute_pc(conjunction)
    assert result.method == "hall"
    assert result.details["pc_2d"] == foster(conjunction.plane, 5.0).value
    assert result.diagnostics[0].startswith("2-D Pc assumptions questionable")


def test_auto_falls_back_to_monte_carlo(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: Conjunction) -> PcResult:
        raise PcConvergenceError(
            "Hall 3D-Nc: the peak-overlap iteration did not converge"
        )

    calls: list[dict[str, object]] = []

    def run(_: Conjunction, **options: object) -> PcResult:
        calls.append(options)
        return PcResult(0.25, "montecarlo", (), {})

    monkeypatch.setattr(api, "hall3d", fail)
    monkeypatch.setattr(api, "montecarlo", run)
    result = compute_pc(along_track_conjunction(1.5 * DURATION_LIMIT * LEO_PERIOD))
    assert result.method == "montecarlo"
    assert result.diagnostics[-1].endswith(
        "; Monte Carlo used instead (seed 0, at most 1,000,000 trials)"
    )
    # A fixed seed and a bounded budget keep ``auto`` reproducible and bounded.
    assert calls == [{"max_samples": AUTO_MC_SAMPLES, "seed": AUTO_MC_SEED}]


def test_auto_uses_hall_at_zero_relative_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Hall's accuracy at zero relative velocity is checked on Alfano case 12
    # (validation); here only the choice is.
    monkeypatch.setattr(api, "hall3d", lambda c: PcResult(0.5, "hall", (), {}))
    offset = np.array([0.0, 0.0, 10.0])
    primary = object_state(LEO_R, LEO_V, 25.0 * np.eye(3))
    secondary = object_state(LEO_R + offset, LEO_V, 25.0 * np.eye(3))
    result = compute_pc(Conjunction(primary, secondary, 5.0))
    assert result.method == "hall"
    assert result.diagnostics == ("relative velocity is zero: 2-D Pc is undefined",)
    assert "encounter_duration_s" not in result.details


def test_auto_refuses_zero_relative_velocity_with_an_unknown_covariance() -> None:
    # Max-Pc needs the encounter plane and the 3-D methods both covariances.
    primary = object_state(LEO_R, LEO_V, 25.0 * np.eye(3))
    secondary = ObjectState(
        "SEC",
        StateVector(EPOCH, LEO_R + np.array([0.0, 0.0, 10.0]), LEO_V, "GCRF"),
        None,
    )
    conjunction = Conjunction(primary, secondary, 5.0)
    with pytest.raises(
        GeometryError,
        match=r"velocity is zero .* the covariance of secondary \(SEC\) is unknown$",
    ):
        compute_pc(conjunction)
    both = Conjunction(
        ObjectState("PRI", primary.state, None), secondary, conjunction.hard_body_radius
    )
    with pytest.raises(
        GeometryError,
        match=r"velocity is zero .* the covariances of primary \(PRI\) and "
        r"secondary \(SEC\) are unknown$",
    ):
        compute_pc(both)


def test_auto_refuses_zero_relative_velocity_with_an_unbound_orbit() -> None:
    fast = 1.5 * np.asarray(LEO_V)
    primary = object_state(LEO_R, fast, 25.0 * np.eye(3))
    secondary = object_state(LEO_R + np.array([0.0, 0.0, 10.0]), fast, 25.0 * np.eye(3))
    with pytest.raises(
        GeometryError, match=r"relative velocity is zero .* not elliptic"
    ):
        compute_pc(Conjunction(primary, secondary, 5.0))


def test_options_need_an_explicit_method() -> None:
    with pytest.raises(ValueError, match="need an explicit method"):
        compute_pc(head_on(30.0, 10.0, 20.0, 5.0), rtol=1e-6)
