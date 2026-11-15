import math
import operator
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from keepout.risk import PcObservation, RiskLevel, Thresholds


def seen(pc: float, ci_high: float | None = None) -> PcObservation:
    return PcObservation(time_to_tca=3600.0, source="test", pc=pc, ci_high=ci_high)


def classify(observation: PcObservation, limits: Thresholds | None = None) -> RiskLevel:
    limits = limits if limits is not None else Thresholds()
    return limits.level(limits.value_of(observation))


@pytest.mark.parametrize(
    ("pc", "level"),
    [
        (0.0, RiskLevel.GREEN),
        (9.99e-8, RiskLevel.GREEN),
        (1e-7, RiskLevel.YELLOW),
        (9.99e-5, RiskLevel.YELLOW),
        (1e-4, RiskLevel.RED),
        (1.0, RiskLevel.RED),
    ],
)
def test_the_default_thresholds_give_each_level(pc: float, level: RiskLevel) -> None:
    assert classify(seen(pc)) is level


def test_levels_are_ordered_by_severity() -> None:
    assert RiskLevel.GREEN < RiskLevel.YELLOW < RiskLevel.RED
    assert not isinstance(RiskLevel.GREEN, int)


def test_levels_do_not_compare_with_other_types() -> None:
    with pytest.raises(TypeError, match="not supported"):
        operator.lt(RiskLevel.GREEN, 1)


def test_conservative_basis_uses_the_upper_end_of_an_interval() -> None:
    observation = seen(5e-5, ci_high=2e-4)
    assert classify(observation) is RiskLevel.RED
    assert classify(observation, Thresholds(basis="estimate")) is RiskLevel.YELLOW


def test_custom_thresholds_move_the_level_boundaries() -> None:
    limits = Thresholds(green_below=1e-6, red_from=1e-3)
    assert classify(seen(5e-7), limits) is RiskLevel.GREEN
    assert classify(seen(1e-6), limits) is RiskLevel.YELLOW
    assert classify(seen(5e-4), limits) is RiskLevel.YELLOW
    # Between the default red_from (1e-4) and the custom one.
    assert classify(seen(9.99e-4), limits) is RiskLevel.YELLOW
    assert classify(seen(1e-3), limits) is RiskLevel.RED


@pytest.mark.parametrize(
    "changes",
    [
        {"green_below": 0.0},
        {"green_below": 1e-4},
        {"green_below": 1e-3},
        {"red_from": 1.5},
        {"red_from": math.inf},
        {"green_below": math.nan},
    ],
)
def test_invalid_thresholds_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="0 < green_below < red_from <= 1"):
        Thresholds(**changes)


def test_unknown_basis_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported basis"):
        Thresholds(basis="optimistic")  # type: ignore[arg-type]


def test_an_observation_without_pc_has_no_level() -> None:
    failed = PcObservation(time_to_tca=0.0, source="test", error="no HBR")
    with pytest.raises(ValueError, match="observation has no Pc: no HBR"):
        classify(failed)


@given(
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.sampled_from(["conservative", "estimate"]),
)
def test_a_higher_pc_is_never_a_lower_level(
    low: float, high: float, basis: Any
) -> None:
    low, high = sorted((low, high))
    limits = Thresholds(basis=basis)
    assert classify(seen(low), limits) <= classify(seen(high), limits)


def test_a_level_has_a_stable_key_for_data() -> None:
    assert [level.key for level in RiskLevel] == ["green", "yellow", "red"]
