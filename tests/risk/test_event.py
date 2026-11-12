import itertools
import math

import pytest
from builders import CDM_DATA
from risk_builders import b1_cdm

import keepout.cdm
from keepout import EventError, PcConvergenceError, compute_pc
from keepout.cdm import Cdm
from keepout.events import (
    EVENT_ID_LENGTH,
    ConjunctionEvent,
    EventBounds,
    group_events,
    missing_hard_body_radius,
    stable_event_id,
    time_to_tca,
    without_hard_body_radius,
)
from keepout.pc import MethodChoice, api, computation
from keepout.risk import Action, assess
from keepout.units import DAY

HBR = 20.0
B1_TIME_TO_TCA = DAY + 6 * 60 + 40.618
"""From the annex example's creation to its TCA [s]."""


def test_cdms_are_kept_oldest_first() -> None:
    late = b1_cdm("B", created="2010-03-13T10:00:00.000")
    early = b1_cdm("A", created="2010-03-12T10:00:00.000")
    event = ConjunctionEvent("E", (late, early))
    assert [cdm.message_id for cdm in event.cdms] == ["A", "B"]
    assert event.latest is late
    assert (event.primary_id, event.secondary_id) == ("12345", "30337")


def test_simultaneous_cdms_are_ordered_by_message_id() -> None:
    event = ConjunctionEvent("E", (b1_cdm("B"), b1_cdm("A")))
    assert [cdm.message_id for cdm in event.cdms] == ["A", "B"]


def test_time_to_tca_runs_from_creation_to_tca() -> None:
    assert time_to_tca(b1_cdm()) == pytest.approx(B1_TIME_TO_TCA, abs=1e-6)
    after = b1_cdm(created="2010-03-13T23:37:52.618")
    assert time_to_tca(after) == pytest.approx(-3600.0, abs=1e-6)


def test_event_id_defaults_to_the_stable_id_of_the_first_cdm() -> None:
    # The first by CREATION_DATE, then MESSAGE_ID; a CONJUNCTION_ID changes
    # nothing.
    first = b1_cdm("B", created="2010-03-12T00:00:00.000")
    cdms = [b1_cdm("A", conjunction_id="EVT-9"), first, b1_cdm("C")]
    assert ConjunctionEvent.from_cdms(cdms).event_id == stable_event_id("B")
    tied = [b1_cdm("D"), b1_cdm("C")]
    assert ConjunctionEvent.from_cdms(tied).event_id == stable_event_id("C")


def test_a_stable_event_id_is_twelve_hex_characters_of_the_sha256() -> None:
    assert stable_event_id("abc") == "ba7816bf8f01"
    assert len(stable_event_id("B1-1")) == EVENT_ID_LENGTH == 12


def test_an_explicit_event_id_wins() -> None:
    event = ConjunctionEvent.from_cdms([b1_cdm(conjunction_id="EVT-9")], event_id="X")
    assert event.event_id == "X"


@pytest.mark.parametrize(
    ("cdms", "tolerance", "message"),
    [
        ((), 600.0, "at least one CDM"),
        ((b1_cdm("A"), b1_cdm("A")), 600.0, "MESSAGE_ID repeated: A"),
        (
            (b1_cdm("A"), b1_cdm("B", secondary="40000")),
            600.0,
            "must share both objects, got 12345/30337, 12345/40000",
        ),
        (
            (b1_cdm("A"), b1_cdm("B", tca="2010-03-13T22:48:00.000")),
            600.0,
            "TCAs spread over 607 s, more than tca_tolerance",
        ),
        (
            (b1_cdm("A", conjunction_id="X"), b1_cdm("B", conjunction_id="Y")),
            600.0,
            "different CONJUNCTION_ID values: X, Y",
        ),
        ((b1_cdm(),), -1.0, "tca_tolerance must be finite and >= 0"),
        ((b1_cdm(),), math.nan, "tca_tolerance must be finite and >= 0"),
    ],
)
def test_invalid_events_are_refused(
    cdms: tuple[keepout.cdm.Cdm, ...], tolerance: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ConjunctionEvent.from_cdms(cdms, tca_tolerance=tolerance)


def test_cdms_that_do_not_make_an_event_are_a_keepout_error() -> None:
    with pytest.raises(EventError, match="MESSAGE_ID repeated: A"):
        ConjunctionEvent.from_cdms((b1_cdm("A"), b1_cdm("A")))


def test_an_empty_event_id_is_refused() -> None:
    with pytest.raises(ValueError, match="event_id must not be empty"):
        ConjunctionEvent("", (b1_cdm(),))


def test_observations_are_keepouts_pc_of_each_cdm() -> None:
    first = b1_cdm("A", created="2010-03-11T22:37:52.618")
    second = b1_cdm("B")
    seen = ConjunctionEvent("E", (second, first)).observations(
        combined_hard_body_radius=HBR
    )
    expected = [
        compute_pc(cdm.conjunction(combined_hard_body_radius=HBR)).value
        for cdm in (first, second)
    ]
    assert [o.message_id for o in seen] == ["A", "B"]
    assert [o.pc for o in seen] == expected
    assert seen[0].source == "keepout (foster)"  # what "auto" chose
    assert seen[0].time_to_tca == pytest.approx(2 * DAY, abs=1e-6)
    assert seen[1].time_to_tca == pytest.approx(B1_TIME_TO_TCA, abs=1e-6)
    assert seen[0].reported_pc == 4.835e-05


def test_an_assumed_covariance_is_flagged_on_its_observation_and_event() -> None:
    measured = b1_cdm("A", created="2010-03-11T22:37:52.618")
    assumed = b1_cdm("B", assumed_covariance=True)
    event = ConjunctionEvent("E", (measured, assumed))
    seen = event.observations(combined_hard_body_radius=HBR)
    assert [o.assumed_covariance for o in seen] == [False, True]
    assert event.assumed_covariance
    assert not ConjunctionEvent("F", (measured,)).assumed_covariance


def test_a_named_method_is_used() -> None:
    (seen,) = ConjunctionEvent("E", (b1_cdm(),)).observations(
        compute=computation("elrod"), combined_hard_body_radius=HBR
    )
    assert seen.source == "keepout (elrod)"
    assert seen.result is not None
    assert seen.result.method == "elrod"


def test_a_computation_with_options_is_used() -> None:
    (seen,) = ConjunctionEvent("E", (b1_cdm(),)).observations(
        compute=computation("montecarlo", samples=2_000, seed=1),
        combined_hard_body_radius=HBR,
    )
    assert seen.source == "keepout (montecarlo)"
    assert seen.ci_high is not None


@pytest.mark.parametrize(
    ("method", "source"), [("auto", "keepout (auto)"), ("hall3d", "keepout (hall3d)")]
)
def test_a_failed_cdm_is_kept_with_its_error(
    monkeypatch: pytest.MonkeyPatch, method: MethodChoice, source: str
) -> None:
    # A failure is labelled by the method asked for, since none gave a value.
    def stuck(*_: object) -> keepout.PcResult:
        raise PcConvergenceError("rate not computable")

    monkeypatch.setattr(api, "_compute", stuck)
    (seen,) = ConjunctionEvent("E", (b1_cdm(),)).observations(
        compute=computation(method), combined_hard_body_radius=HBR
    )
    assert not seen.ok
    assert seen.error == "PcConvergenceError: rate not computable"
    assert (seen.source, seen.message_id, seen.reported_pc) == (
        source,
        "B1-1",
        4.835e-05,
    )


def test_a_missing_hbr_is_reported_before_any_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_: object) -> keepout.PcResult:
        raise AssertionError("computed")

    monkeypatch.setattr(api, "_compute", refuse)
    event = ConjunctionEvent("E", (b1_cdm("A"), b1_cdm("B")))
    with pytest.raises(
        ValueError, match="CDM A gives no HBR for OBJECT1 and OBJECT2; CDM B gives none"
    ):
        event.observations()


def test_a_cdm_without_hbr_is_named() -> None:
    message = keepout.cdm.load(CDM_DATA / "tracss_v2.kvn")
    with pytest.raises(ValueError, match="TRACSS-TEST-0001"):
        ConjunctionEvent("E", (message,)).observations(compute=computation("foster"))


def test_hbr_from_the_cdms() -> None:
    text = (CDM_DATA / "tracss_v2.kvn").read_text(encoding="utf-8")
    secondary = "MANEUVERABLE                   = NO\n"
    assert text.count(secondary) == 1
    message = keepout.cdm.loads(
        text.replace(
            secondary, f"{secondary}HBR                            = 1.00 [m]\n"
        )
    )
    (seen,) = ConjunctionEvent("E", (message,)).observations(
        compute=computation("foster")
    )
    conjunction = message.conjunction()
    assert conjunction.combined_hard_body_radius == 6.0
    assert seen.pc == compute_pc(conjunction, "foster").value


def test_an_invalid_reported_pc_is_noted_not_used() -> None:
    (seen,) = ConjunctionEvent("E", (b1_cdm(reported_pc="2.0"),)).observations(
        combined_hard_body_radius=HBR
    )
    assert seen.reported_pc is None
    assert seen.diagnostics == ("reported COLLISION_PROBABILITY 2.0 is not in [0, 1]",)


def test_an_event_is_assessed_from_its_observations() -> None:
    event = ConjunctionEvent("E", (b1_cdm(),))
    assessment = assess(event.observations(combined_hard_body_radius=HBR))
    assert assessment.decision.action is Action.MONITOR  # 4.7e-7: yellow


def test_group_events_splits_by_objects_and_encounter() -> None:
    cdms = [
        b1_cdm("A2", created="2010-03-13T00:00:00.000", tca="2010-03-13T22:37:53.000"),
        b1_cdm("B1", secondary="40000"),
        b1_cdm("A1"),
        b1_cdm("A3", tca="2010-03-14T00:10:00.000"),  # one orbit later
    ]
    events = group_events(cdms)
    assert [[c.message_id for c in e.cdms] for e in events] == [
        ["A1", "A2"],
        ["B1"],
        ["A3"],
    ]
    assert [e.event_id for e in events] == [
        stable_event_id("A1"),
        stable_event_id("B1"),
        stable_event_id("A3"),
    ]


def test_group_events_anchors_the_tolerance_at_the_first_tca() -> None:
    cdms = [
        b1_cdm("A", tca="2010-03-13T22:30:00.000"),
        b1_cdm("B", tca="2010-03-13T22:38:00.000"),
        b1_cdm("C", tca="2010-03-13T22:46:00.000"),
    ]
    assert [len(e.cdms) for e in group_events(cdms)] == [2, 1]


def test_group_events_does_not_depend_on_the_input_order() -> None:
    cdms = [
        b1_cdm("A2", created="2010-03-13T00:00:00.000", tca="2010-03-13T22:37:53.000"),
        b1_cdm("B1", secondary="40000"),
        b1_cdm("A1"),
        b1_cdm("A3", tca="2010-03-14T00:10:00.000"),
    ]

    def grouped(order: tuple[Cdm, ...]) -> list[tuple[str, list[str]]]:
        return [
            (e.event_id, [c.message_id for c in e.cdms]) for e in group_events(order)
        ]

    expected = grouped(tuple(cdms))
    for order in itertools.permutations(cdms):
        assert grouped(order) == expected


def test_group_events_keeps_tcas_exactly_the_tolerance_apart_together() -> None:
    cdms = [
        b1_cdm("A", tca="2010-03-13T22:30:00.000"),
        b1_cdm("B", tca="2010-03-13T22:40:00.000"),
    ]
    assert [len(e.cdms) for e in group_events(cdms, tca_tolerance=600.0)] == [2]


def test_group_events_keeps_different_conjunction_ids_apart() -> None:
    cdms = [
        b1_cdm("A", conjunction_id="C1"),
        b1_cdm("B", conjunction_id="C2"),
        b1_cdm("C", tca="2010-03-13T22:38:00.000", conjunction_id="C1"),
    ]
    events = group_events(cdms)
    assert [[c.message_id for c in e.cdms] for e in events] == [["A", "C"], ["B"]]


def test_a_cdm_without_conjunction_id_between_two_events_is_refused() -> None:
    cdms = [
        b1_cdm("A", conjunction_id="C1"),
        b1_cdm("B", conjunction_id="C2"),
        b1_cdm("C", tca="2010-03-13T22:38:00.000"),
    ]
    with pytest.raises(EventError, match="CDM C fits more than one event"):
        group_events(cdms)


def test_bounds_admit_a_cdm_by_objects_tca_and_conjunction_id() -> None:
    bounds = EventBounds.of(b1_cdm("A", conjunction_id="C1"))
    assert bounds.admits(b1_cdm("B", tca="2010-03-13T22:40:00.000"))
    assert not bounds.admits(b1_cdm("B", tca="2010-03-13T22:50:00.000"))
    assert not bounds.admits(b1_cdm("B", conjunction_id="C2"))
    assert not bounds.admits(b1_cdm("B", secondary="40000"))
    wider = bounds.including(b1_cdm("B", tca="2010-03-13T22:30:00.000"))
    assert wider.conjunction_id == "C1"
    assert wider.spread == pytest.approx(472.618, abs=1e-3)
    assert not wider.admits(b1_cdm("C", tca="2010-03-13T22:42:00.000"))


def test_group_events_of_nothing_is_empty() -> None:
    assert group_events([]) == []


def test_group_events_refuses_an_invalid_tolerance() -> None:
    with pytest.raises(ValueError, match="tca_tolerance must be finite and >= 0"):
        group_events([b1_cdm()], tca_tolerance=-5.0)


def test_the_cdms_without_hbr_are_listed_by_message_id() -> None:
    text = (CDM_DATA / "tracss_v2.kvn").read_text(encoding="utf-8")
    secondary = "MANEUVERABLE                   = NO\n"
    with_hbr = keepout.cdm.loads(
        text.replace(
            secondary, f"{secondary}HBR                            = 1.00 [m]\n"
        )
    )
    cdms = [b1_cdm("A"), with_hbr, b1_cdm("B")]
    assert without_hard_body_radius(cdms) == ("A", "B")
    assert without_hard_body_radius([with_hbr]) == ()
    assert missing_hard_body_radius([*cdms, keepout.cdm.loads(text)]) == (
        "CDM A gives no HBR for OBJECT1 and OBJECT2; CDM B gives none for OBJECT1 "
        "and OBJECT2; CDM TRACSS-TEST-0001 gives none for OBJECT2"
    )
    assert missing_hard_body_radius([with_hbr]) == ""
