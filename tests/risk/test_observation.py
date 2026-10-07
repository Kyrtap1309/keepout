import math
from typing import Any

import pytest

from keepout.pc.result import PcResult
from keepout.risk import DAY, PcObservation


def observation(**changes: Any) -> PcObservation:
    fields: dict[str, Any] = {"time_to_tca": DAY, "source": "test", "pc": 1e-5}
    fields.update(changes)
    return PcObservation(**fields)


def test_an_observation_keeps_its_values() -> None:
    seen = observation(message_id="M1", reported_pc=2e-5, diagnostics=["note"])
    assert seen.ok
    assert (seen.pc, seen.kind, seen.ci_high, seen.error) == (
        1e-5,
        "estimate",
        None,
        None,
    )
    assert (seen.message_id, seen.reported_pc, seen.diagnostics) == (
        "M1",
        2e-5,
        ("note",),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"pc": 0.0},
        {"pc": 1.0},
        {"pc": 1e-5, "ci_high": 1e-5},
        {"pc": 0.5, "ci_high": 1.0},
    ],
    ids=["zero", "one", "interval-end-at-pc", "interval-end-at-one"],
)
def test_edge_probabilities_are_accepted(changes: dict[str, Any]) -> None:
    assert observation(**changes).ok


def test_a_negative_zero_is_stored_as_zero() -> None:
    # Else an observation at TCA sorts and prints by the sign of its zero.
    seen = observation(time_to_tca=-0.0, pc=-0.0, ci_high=-0.0)
    assert seen.pc is not None
    assert seen.ci_high is not None
    values = (seen.time_to_tca, seen.pc, seen.ci_high)
    assert [math.copysign(1.0, value) for value in values] == [1.0, 1.0, 1.0]


def test_a_failed_observation_has_an_error_and_no_pc() -> None:
    failed = observation(pc=None, error="PcConvergenceError: no convergence")
    assert not failed.ok
    assert failed.pc is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"time_to_tca": math.nan}, "time_to_tca must be finite"),
        ({"source": ""}, "source must not be empty"),
        ({"pc": None}, "exactly one of pc and error"),
        ({"error": "both"}, "exactly one of pc and error"),
        ({"pc": None, "error": ""}, "error must not be empty"),
        ({"pc": 1.5}, r"pc must be in \[0, 1\]"),
        ({"pc": math.nan}, r"pc must be in \[0, 1\]"),
        ({"ci_high": 1e-6}, "must not be below pc"),
        ({"ci_high": 2.0}, r"ci_high must be in \[0, 1\]"),
        ({"pc": None, "error": "x", "ci_high": 1e-3}, "ci_high needs pc"),
        ({"reported_pc": -1e-3}, r"reported_pc must be in \[0, 1\]"),
        ({"kind": "guess"}, "kind must be one of"),
    ],
)
def test_invalid_observations_are_refused(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        observation(**changes)


def test_an_estimate_from_a_keepout_result() -> None:
    result = PcResult(3e-5, "foster", ("primary (1): covariance repaired",))
    seen = PcObservation.from_result(
        result, time_to_tca=2 * DAY, message_id="M2", reported_pc=4e-5
    )
    assert (seen.pc, seen.kind, seen.ci_high) == (3e-5, "estimate", None)
    assert seen.source == "keepout (foster)"
    assert seen.diagnostics == result.diagnostics
    assert (seen.time_to_tca, seen.message_id, seen.reported_pc) == (
        2 * DAY,
        "M2",
        4e-5,
    )
    assert seen.result is result


@pytest.mark.parametrize("method", ["frisbee", "dilution", "alfano"])
def test_a_maximum_pc_is_an_upper_bound(method: Any) -> None:
    seen = PcObservation.from_result(PcResult(1e-3, method), time_to_tca=DAY)
    assert seen.kind == "upper_bound"


def test_monte_carlo_keeps_the_upper_end_of_its_interval() -> None:
    result = PcResult(2e-4, "montecarlo", details={"ci_low": 1e-4, "ci_high": 3e-4})
    seen = PcObservation.from_result(result, time_to_tca=DAY)
    assert (seen.kind, seen.ci_high) == ("estimate", 3e-4)


def test_only_monte_carlo_has_an_interval() -> None:
    result = PcResult(2e-4, "hall3d", details={"nc": 2e-4})
    assert PcObservation.from_result(result, time_to_tca=DAY).ci_high is None
