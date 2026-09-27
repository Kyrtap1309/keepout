import numpy as np
import pytest
from pc_builders import LEO_R, LEO_V, head_on, object_state

from keepout.conjunction import Conjunction
from keepout.pc import compute_pc, elrod, foster
from keepout.pc.api import encounter_duration, orbital_period


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
    assert result.details["order"] >= 64


def test_unknown_method_is_rejected() -> None:
    conjunction = head_on(30.0, 10.0, 20.0, 5.0)
    with pytest.raises(ValueError, match="unknown Pc method 'hall'"):
        compute_pc(conjunction, "hall")  # type: ignore[arg-type]


def test_fast_leo_encounter_has_no_warnings() -> None:
    result = compute_pc(head_on(30.0, 10.0, 20.0, 5.0))
    assert result.diagnostics == ()
    # 2 (5 * 1 m + 5 m) / 15092 m/s: the primary's y-sigma lies along v_rel.
    assert result.details["encounter_duration_s"] == pytest.approx(
        2 * 10.0 / 15092.0, rel=1e-6
    )


def test_conjunction_diagnostics_come_first() -> None:
    base = head_on(30.0, 10.0, 20.0, 5.0)
    conjunction = Conjunction(base.primary, base.secondary, 5.0, ("OBJECT1: repaired",))
    assert compute_pc(conjunction).diagnostics == ("OBJECT1: repaired",)


def test_slow_encounter_warns_about_2d_assumptions() -> None:
    # 1 m/s relative speed, 100 m sigma: ~1000 s, about 17% of a LEO period.
    primary = object_state(LEO_R, LEO_V, 1e4 * np.eye(3))
    secondary = object_state(
        LEO_R + np.array([20.0, 0.0, 0.0]), LEO_V + np.array([0.0, 0.0, 1.0]), np.eye(3)
    )
    result = compute_pc(Conjunction(primary, secondary, 5.0))
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].startswith("2-D Pc assumptions questionable")


def test_states_away_from_tca_are_reported() -> None:
    # Secondary 10 m along the relative velocity from closest approach.
    primary = object_state(LEO_R, LEO_V, 100.0 * np.eye(3))
    secondary = object_state(LEO_R + np.array([30.0, -10.0, 0.0]), -LEO_V, np.eye(3))
    result = compute_pc(Conjunction(primary, secondary, 5.0))
    assert result.diagnostics == (
        "states are 10 m from closest approach along the relative velocity; "
        "Pc uses the straight-line closest approach",
    )


def test_orbital_period_of_a_circular_leo() -> None:
    primary = object_state(LEO_R, LEO_V, np.eye(3))
    circular_speed = np.sqrt(3.986004418e14 / 7000e3)
    period = orbital_period(primary)
    assert period is not None
    # LEO_V is within 0.03% of circular: period within ~0.1% of 2 pi r / v.
    assert period == pytest.approx(2 * np.pi * 7000e3 / circular_speed, rel=2e-3)


def test_unbound_orbit_has_no_period() -> None:
    assert orbital_period(object_state(LEO_R, 3 * LEO_V, np.eye(3))) is None


def test_encounter_duration_uses_the_sigma_along_the_relative_velocity() -> None:
    # head_on: relative velocity along GCRF -y, primary y-variance 1 m^2,
    # secondary ~0: the extent along v_rel is 1 m, whatever the plane sigmas.
    conjunction = head_on(30.0, 10.0, 40.0, 5.0)
    assert encounter_duration(conjunction) == pytest.approx(
        2 * (5 * 1.0 + 5.0) / conjunction.relative_speed, rel=1e-6
    )


def test_large_sigma_across_a_slow_track_does_not_warn() -> None:
    # 20 m/s along z; 2 km sigma along x but 1 m along z: crossing ~0.5 s.
    primary = object_state(LEO_R, LEO_V, np.diag([4e6, 4e6, 1.0]))
    secondary = object_state(
        LEO_R + np.array([20.0, 0.0, 0.0]),
        LEO_V + np.array([0.0, 0.0, 20.0]),
        np.eye(3),
    )
    assert compute_pc(Conjunction(primary, secondary, 5.0)).diagnostics == ()
