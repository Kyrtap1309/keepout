import dataclasses
import json
from typing import Any, get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st

import keepout
from keepout.errors import StoreError
from keepout.pc import Method, PcResult, computation
from keepout.risk import PcObservation, Thresholds
from keepout.risk.observation import PC_KINDS
from keepout.settings import Settings
from keepout.store.codec import observation_from_json, observation_to_json, result_key
from keepout.units import DAY

_FIELDS = tuple(
    field.name for field in dataclasses.fields(PcObservation) if field.name != "result"
)
_TEXT = st.text(min_size=1, max_size=20)
_PROBABILITY = st.floats(min_value=0.0, max_value=1.0)


@st.composite
def observations(draw: st.DrawFn) -> PcObservation:
    """A valid observation, with or without a Pc, and no full result."""
    common: dict[str, Any] = {
        "time_to_tca": draw(st.floats(min_value=-10 * DAY, max_value=10 * DAY)),
        "source": draw(_TEXT),
        "kind": draw(st.sampled_from(PC_KINDS)),
        "message_id": draw(st.none() | _TEXT),
        "reported_pc": draw(st.none() | _PROBABILITY),
        "diagnostics": tuple(draw(st.lists(st.text(max_size=20), max_size=3))),
        "assumed_covariance": draw(st.booleans()),
        "method": draw(st.none() | st.sampled_from(get_args(Method))),
    }
    if draw(st.booleans()):
        return PcObservation(error=draw(_TEXT), **common)
    pc = draw(_PROBABILITY)
    ci_high = draw(st.none() | st.floats(min_value=pc, max_value=1.0))
    return PcObservation(pc=pc, ci_high=ci_high, **common)


def _values(observation: PcObservation) -> tuple[object, ...]:
    return tuple(getattr(observation, name) for name in _FIELDS)


@given(observations())
def test_observation_round_trips(observation: PcObservation) -> None:
    payload = json.loads(json.dumps(observation_to_json(observation)))
    back = observation_from_json(payload)
    assert _values(back) == _values(observation)
    assert back.result is None


def test_an_observation_of_a_result_keeps_its_method_not_the_result() -> None:
    result = PcResult(2e-4, "montecarlo", interval=(1e-4, 3e-4))
    seen = PcObservation.from_result(result, time_to_tca=DAY, message_id="M1")
    back = observation_from_json(observation_to_json(seen))
    assert (back.method, back.source, back.ci_high) == (
        "montecarlo",
        "keepout (montecarlo)",
        3e-4,
    )
    assert back.result is None


def _payload(**changes: object) -> dict[str, object]:
    seen = PcObservation(time_to_tca=DAY, source="test", pc=1e-5)
    payload: dict[str, object] = dict(observation_to_json(seen))
    payload.update(changes)
    return payload


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({k: v for k, v in _payload().items() if k != "pc"}, "missing key.*pc"),
        (_payload(extra=1), "unknown key.*extra"),
        (_payload(pc="1e-5"), "pc"),
        (_payload(pc=True), "pc"),
        (_payload(time_to_tca=None), "time_to_tca"),
        (_payload(source=3), "source"),
        (_payload(diagnostics="note"), "diagnostics"),
        (_payload(diagnostics=[1]), "diagnostics"),
        (_payload(assumed_covariance=0), "assumed_covariance"),
        (_payload(method=1), "method"),
        (_payload(pc=2.0), "pc"),
        (_payload(kind="maybe"), "kind"),
        (_payload(error="failed"), "exactly one of pc and error"),
    ],
)
def test_a_malformed_payload_raises_store_error(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(StoreError, match=message):
        observation_from_json(payload)


def test_result_key_changes_with_each_parameter_and_the_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def key(**changes: Any) -> str:
        return result_key(Settings(**changes))

    base = key()
    assert len(base) == 64
    assert int(base, 16) >= 0
    assert key() == base
    others = [
        key(compute=computation("foster")),
        key(compute=computation("montecarlo", samples=10_000, seed=1)),
        key(compute=computation("montecarlo", samples=10_000, seed=2)),
        key(policy="raise"),
        key(combined_hard_body_radius=20.0),
    ]
    assert len({base, *others}) == len(others) + 1
    # The thresholds and the target are applied on read: same cache.
    assert key(thresholds=Thresholds(1e-6, 1e-3), target_pc=1e-6) == base
    monkeypatch.setattr(keepout, "__version__", "0.0.0-other", raising=False)
    assert key() != base


def test_a_radius_keys_the_same_as_an_integer_or_a_float() -> None:
    keys = {
        result_key(Settings(combined_hard_body_radius=radius)) for radius in (20, 20.0)
    }
    assert len(keys) == 1


def test_an_unknown_method_is_a_store_error() -> None:
    data = observation_to_json(PcObservation(DAY, "keepout (foster)", pc=1e-5))
    with pytest.raises(StoreError, match="method"):
        observation_from_json({**data, "method": "nope"})
