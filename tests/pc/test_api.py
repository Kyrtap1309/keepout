import math

import numpy as np
import pytest
from builders import EPOCH, LEO_R, LEO_V, head_on, object_state

from keepout._types import StateVector
from keepout.conjunction import Conjunction
from keepout.covariance import make_covariance
from keepout.objects import ObjectState
from keepout.orbits import orbital_period
from keepout.pc import compute_pc, elrod, foster
from keepout.pc.api import DURATION_LIMIT, SIGMA_EXTENT, encounter_duration


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
    with pytest.raises(ValueError, match="unknown Pc method 'hall'"):
        compute_pc(conjunction, "hall")  # type: ignore[arg-type]


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
    asymmetric = base.primary.covariance.matrix.copy()
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
    period = orbital_period(StateVector(EPOCH, LEO_R, LEO_V, "GCRF"))
    assert period is not None
    conjunction = along_track_conjunction(fraction * DURATION_LIMIT * period)
    assert encounter_duration(conjunction) == pytest.approx(
        fraction * DURATION_LIMIT * period, rel=1e-9
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


def test_states_away_from_tca_are_reported() -> None:
    # Secondary 10 m along the relative velocity from closest approach.
    primary = object_state(LEO_R, LEO_V, 100.0 * np.eye(3))
    secondary = object_state(LEO_R + np.array([30.0, -10.0, 0.0]), -LEO_V, np.eye(3))
    (message,) = compute_pc(Conjunction(primary, secondary, 5.0)).diagnostics
    assert message.startswith("states are 10 m from closest approach")


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
