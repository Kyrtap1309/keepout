"""Pc of real approaches with the synthetic covariance: every method takes
its zero velocity block, and ``auto`` keeps the 2-D Pc."""

import pytest
from screening_data import load_fixture

from keepout._units import DAY
from keepout.conjunction import Conjunction
from keepout.pc import compute_pc
from keepout.screening import CloseApproach, screen
from keepout.synthetic import SyntheticCovarianceModel

BY_NUMBER = {omm.catalog_number: omm for omm in load_fixture()}
START = max(omm.epoch for omm in BY_NUMBER.values())
MODEL = SyntheticCovarianceModel()


def first_approach(primary: int, secondary: int, days: float) -> CloseApproach:
    result = screen([BY_NUMBER[primary]], [BY_NUMBER[secondary]], START, days * DAY)
    return result.approaches[0]


def with_model(approach: CloseApproach) -> Conjunction:
    primary = BY_NUMBER[approach.primary_number]
    secondary = BY_NUMBER[approach.secondary_number]
    return approach.to_conjunction(
        hard_body_radius=10.0,
        primary_covariance=MODEL.covariance(primary, approach.tca),
        secondary_covariance=MODEL.covariance(secondary, approach.tca),
    )


def test_a_fast_approach_takes_every_method() -> None:
    # Two LEO objects crossing at 4.9 km/s, 1.1 km apart.
    conjunction = with_model(first_approach(52939, 57087, 5.0))
    assert conjunction.relative_speed == pytest.approx(4906.4, abs=0.1)
    auto = compute_pc(conjunction)
    assert auto.method == "foster"
    assert auto.diagnostics == ()
    assert auto.value == pytest.approx(1.5057e-8, rel=1e-4)
    assert compute_pc(conjunction, "elrod").value == pytest.approx(auto.value, rel=1e-9)
    # A short encounter: the 3-D Pc agrees with the 2-D one.
    assert compute_pc(conjunction, "hall3d").value == pytest.approx(
        auto.value, rel=5e-3
    )
    sampled = compute_pc(conjunction, "montecarlo", samples=1_000, seed=1)
    # Pc ~ 1.5e-8 gives no hit in 1000 trials; the interval still holds it.
    assert sampled.value == 0.0
    assert sampled.details["samples"] == 1_000
    assert sampled.details["ci_high"] > auto.value


def test_a_slow_approach_keeps_the_2d_pc_with_both_notes() -> None:
    # Two objects of one launch drifting 0.26 m/s apart, 200 m away: the
    # 2-D assumptions fail, but the covariances have no velocity block.
    conjunction = with_model(first_approach(62079, 62080, 1.0))
    assert conjunction.relative_speed == pytest.approx(0.255, abs=1e-3)
    auto = compute_pc(conjunction)
    assert auto.method == "foster"
    assert auto.value == pytest.approx(2.1871e-5, rel=1e-4)
    questionable, position_only = auto.diagnostics
    assert questionable.startswith("2-D Pc assumptions questionable")
    assert position_only == (
        "the covariances of primary (62079) and secondary (62080) are known "
        "only in position (zero velocity block): the 3-D methods need the "
        "velocity uncertainty, so the 2-D Pc is used"
    )
    assert compute_pc(conjunction, "elrod").value == pytest.approx(auto.value, rel=1e-9)
