import math
from random import Random
from typing import Any

import pytest
from builders import at, series
from hypothesis import given
from hypothesis import strategies as st
from risk_builders import observations, pcs

from keepout.risk import (
    Action,
    DecisionConfig,
    PcKind,
    PcObservation,
    RiskLevel,
    Thresholds,
    analyse_trend,
    assess,
    decide,
)
from keepout.units import DAY


def decision_for(seen: list[PcObservation], **config: float) -> Action:
    return decide(analyse_trend(seen), config=DecisionConfig(**config)).action


@pytest.mark.parametrize(
    ("seen", "expected"),
    [
        (series((3.0, 1e-9), (1.0, 1e-9)), Action.IGNORE),
        (series((3.0, 1e-6), (1.0, 1e-6)), Action.MONITOR),
        (series((3.0, 1e-3), (1.0, 1e-3)), Action.MANEUVER),
        (series((4.0, 1e-3), (3.0, 1e-3)), Action.PREPARE_MANEUVER),
        (series((3.0, 1e-3), (2.0, 1e-3)), Action.MANEUVER),
        (series((3.0, 1e-12), (2.0, 1e-10), (1.0, 1e-8)), Action.MONITOR),
        (series((3.0, 1e-8), (2.0, 1e-10), (1.0, 1e-12)), Action.IGNORE),
        (series((3.0, 1e-18), (2.0, 1e-17), (1.0, 1e-16)), Action.IGNORE),
    ],
    ids=[
        "green",
        "yellow",
        "red-within-decision-point",
        "red-before-decision-point",
        "red-at-decision-point",
        "green-rising",
        "green-falling",
        "green-rising-far-below",
    ],
)
def test_each_rule_gives_its_action(
    seen: list[PcObservation], expected: Action
) -> None:
    assert decision_for(seen) == expected


def test_a_rising_green_pc_says_whether_it_reaches_yellow() -> None:
    fast = decide(analyse_trend(series((3.0, 1e-12), (2.0, 1e-10), (1.0, 1e-8))))
    assert fast.reasons[-1] == "rising fast enough to reach 1e-07 before TCA"
    slow = decide(analyse_trend(series((3.0, 1e-18), (2.0, 1e-17), (1.0, 1e-16))))
    assert slow.reasons[-1] == "rising, but at this rate below 1e-07 at TCA"


def test_a_rising_green_pc_names_a_custom_threshold_exactly() -> None:
    custom = Thresholds(green_below=2.5e-7, red_from=1e-4)
    fast = analyse_trend(
        series((3.0, 1e-12), (2.0, 1e-10), (1.0, 1e-8)), thresholds=custom
    )
    reasons = decide(fast).reasons
    assert reasons[-1] == "rising fast enough to reach 2.5e-07 before TCA"
    slow = analyse_trend(
        series((3.0, 1e-18), (2.0, 1e-17), (1.0, 1e-16)), thresholds=custom
    )
    reasons = decide(slow).reasons
    assert reasons[-1] == "rising, but at this rate below 2.5e-07 at TCA"


def test_rising_is_judged_against_the_trends_thresholds() -> None:
    seen = series((3.0, 1e-18), (2.0, 1e-17), (1.0, 1e-16))
    assert decide(analyse_trend(seen)).action is Action.IGNORE
    low = Thresholds(green_below=1e-15, red_from=1e-4)
    assert decide(analyse_trend(seen, thresholds=low)).action is Action.MONITOR


def test_decision_point_is_configurable() -> None:
    seen = series((4.0, 1e-3), (3.0, 1e-3))
    assert decision_for(seen, decision_point=3.5 * 86400) == Action.MANEUVER


def test_a_manoeuvre_says_why() -> None:
    decision = decide(analyse_trend(series((3.0, 1e-3), (1.0, 2e-3))))
    assert decision.reasons == (
        "latest Pc 2.00e-03 is red, 1.00 d before TCA",
        "trend steady (+0.15 decades per day)",
        "within the decision point (2.00 d before TCA)",
    )


def test_a_green_upper_bound_is_monitored() -> None:
    bound = at(1.0, 1e-9, kind="upper_bound")
    decision = decide(analyse_trend([bound]))
    assert decision.action is Action.MONITOR
    assert decision.reasons == (
        (
            "latest Pc 1.00e-09 is green (an upper bound: a covariance is unknown), "
            "1.00 d before TCA"
        ),
        "trend undetermined",
        "the Pc is an upper bound, not an estimate",
    )


def test_a_green_pc_with_diagnostics_is_monitored() -> None:
    noted = at(1.0, 1e-9, diagnostics=("secondary (2): covariance repaired",))
    decision = decide(analyse_trend([noted]))
    assert decision.action is Action.MONITOR
    assert decision.reasons[-1] == "diagnostic: secondary (2): covariance repaired"


def test_a_red_upper_bound_names_the_concern() -> None:
    decision = decide(analyse_trend([at(1.0, 1e-3, kind="upper_bound")]))
    assert decision.action is Action.MANEUVER
    assert decision.reasons[-1] == "the Pc is an upper bound, not an estimate"


def test_the_interval_end_is_named_when_it_decides() -> None:
    decision = decide(analyse_trend([at(1.0, 5e-5, ci_high=2e-4)]))
    assert decision.action is Action.MANEUVER
    assert decision.reasons[0] == (
        "latest Pc 2.00e-04 is red (upper end of the Monte Carlo confidence "
        "interval), 1.00 d before TCA"
    )


def test_a_floored_pc_says_so() -> None:
    decision = decide(analyse_trend([at(1.0, 1e-30, kind="below_floor")]))
    assert decision.action is Action.IGNORE
    assert "below the source's reporting floor" in decision.reasons[0]


NO_PC = "latest observation has no Pc: GeometryError: no plane"
RED_THEN = "the latest observation with a Pc was red: Pc 1.00e-03, {} d before TCA"
PREPARE = "before the decision point (2.00 d before TCA): prepare a maneuver"
WITHIN = "within the decision point (2.00 d before TCA)"


@pytest.mark.parametrize(
    ("seen", "action", "reasons"),
    [
        (
            [*series((3.0, 1e-9)), at(1.0, None, error="GeometryError: no plane")],
            Action.MONITOR,
            (NO_PC,),
        ),
        (
            [*series((4.0, 1e-3)), at(3.0, None)],
            Action.PREPARE_MANEUVER,
            (NO_PC, RED_THEN.format("4.00"), "trend undetermined", PREPARE),
        ),
        (
            [*series((1.5, 1e-3)), at(1.0, None)],
            Action.MANEUVER,
            (NO_PC, RED_THEN.format("1.50"), "trend undetermined", WITHIN),
        ),
        (
            [*series((1.0, 1e-3)), at(1.0, None)],
            Action.MANEUVER,
            (NO_PC, RED_THEN.format("1.00"), "trend undetermined", WITHIN),
        ),
        (
            [at(3.0, 1e-3, kind="upper_bound"), at(2.5, None)],
            Action.PREPARE_MANEUVER,
            (
                NO_PC,
                (
                    "the latest observation with a Pc was red: Pc 1.00e-03 (an "
                    "upper bound: a covariance is unknown), 3.00 d before TCA"
                ),
                "trend undetermined",
                PREPARE,
                "the Pc is an upper bound, not an estimate",
            ),
        ),
        (
            [*series((6.0, 1e-3), (3.0, 1e-9), (2.0, 1e-9)), at(1.0, None)],
            Action.MONITOR,
            (NO_PC,),
        ),
    ],
    ids=[
        "after-green-is-monitored",
        "after-early-red-prepares-a-manoeuvre",
        "does-not-downgrade-a-manoeuvre",
        "simultaneous-with-red-is-manoeuvred",
        "after-red-upper-bound-names-the-concern",
        "follows-the-latest-pc-not-an-older-red-one",
    ],
)
def test_a_missing_latest_pc_is_judged_by_the_latest_one_that_exists(
    seen: list[PcObservation], action: Action, reasons: tuple[str, ...]
) -> None:
    decision = decide(analyse_trend(seen))
    assert decision.action is action
    assert decision.reasons == reasons


def test_an_event_past_tca_is_monitored() -> None:
    decision = decide(analyse_trend(series((-0.1, 1e-3))))
    assert decision.action is Action.MONITOR
    assert decision.reasons == ("no observation before TCA: nothing to decide on",)


@pytest.mark.parametrize("seconds", [0.0, -1.0, math.inf, math.nan])
def test_invalid_decision_point_is_refused(seconds: float) -> None:
    with pytest.raises(ValueError, match="decision_point must be finite and > 0"):
        DecisionConfig(decision_point=seconds)


def test_assess_combines_trend_and_decision() -> None:
    assessment = assess(series((1.0, 2e-3), (3.0, 1e-3)))
    assert assessment.level is RiskLevel.RED
    assert assessment.decision.action is Action.MANEUVER
    assert assessment.trend.direction == "steady"
    assert assessment.thresholds == Thresholds()


@pytest.mark.parametrize(
    "doubtful",
    [
        {"kind": "upper_bound"},
        {"diagnostics": ("secondary (2): covariance repaired",)},
    ],
    ids=["upper-bound", "diagnostics"],
)
def test_a_doubtful_simultaneous_pc_counts_as_latest(doubtful: dict[str, Any]) -> None:
    estimate, flagged = at(1.0, 1e-9), at(1.0, 1e-9, **doubtful)
    for seen in ([estimate, flagged], [flagged, estimate]):
        assert assess(seen).decision.action is Action.MONITOR


def test_actions_are_ordered_and_not_numbers() -> None:
    assert Action.IGNORE < Action.MONITOR < Action.PREPARE_MANEUVER
    assert Action.PREPARE_MANEUVER < Action.MANEUVER
    assert not isinstance(Action.IGNORE, int)


def test_assess_passes_its_settings_on() -> None:
    assessment = assess(
        series((3.0, 1e-5), (1.0, 1e-5)),
        thresholds=Thresholds(green_below=1e-4, red_from=1e-2),
    )
    assert assessment.level is RiskLevel.GREEN
    assert assessment.decision.action is Action.IGNORE


def test_an_assessment_without_a_latest_pc_has_no_level() -> None:
    assert assess([at(1.0, None)]).level is None
    assert assess(series((-1.0, 1e-3))).level is None


@given(
    st.lists(observations(), max_size=6),
    st.floats(0.0, 7.0),
    pcs,
    pcs,
    st.sampled_from(["estimate", "upper_bound"]),
)
def test_a_higher_latest_pc_never_gives_a_milder_action(
    history: list[PcObservation], days: float, low: float, high: float, kind: PcKind
) -> None:
    low, high = sorted((low, high))
    earlier = [o for o in history if o.time_to_tca > days * DAY]
    milder = assess([*earlier, at(days, low, kind=kind)]).decision.action
    harsher = assess([*earlier, at(days, high, kind=kind)]).decision.action
    assert milder <= harsher


@given(st.lists(observations(), min_size=1, max_size=6))
def test_doubtful_data_is_never_ignored(history: list[PcObservation]) -> None:
    assessment = assess(history)
    latest = assessment.trend.latest_observation
    doubtful = latest is None or not latest.ok or latest.kind == "upper_bound"
    if doubtful or (latest is not None and latest.diagnostics):
        assert assessment.decision.action is not Action.IGNORE


@given(st.lists(observations(), min_size=1, max_size=6), st.randoms())
def test_the_order_of_the_input_does_not_change_the_decision(
    history: list[PcObservation], random: Random
) -> None:
    shuffled = list(history)
    random.shuffle(shuffled)
    first, second = assess(history).decision, assess(shuffled).decision
    assert (first.action, first.reasons) == (second.action, second.reasons)


def test_an_action_is_labelled_in_words() -> None:
    assert [action.label for action in Action] == [
        "ignore",
        "monitor",
        "prepare maneuver",
        "maneuver",
    ]


def test_an_action_has_a_stable_key_for_data() -> None:
    assert [action.key for action in Action] == [
        "ignore",
        "monitor",
        "prepare_maneuver",
        "maneuver",
    ]
