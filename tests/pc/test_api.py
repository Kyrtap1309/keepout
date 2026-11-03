import inspect
import math
import pickle
import re
import typing
from collections.abc import Callable

import numpy as np
import pytest
from builders import (
    EPOCH,
    LEO_R,
    LEO_V,
    TYPICAL_HEAD_ON,
    approx,
    head_on,
    known,
    object_state,
    shifted_back,
    unbound_crossing,
    without_covariance,
)
from faults import (
    auto_hall_integration,
    auto_monte_carlo,
    auto_monte_carlo_budget,
    overlap_failing_while_integrating,
)
from pc_builders import side_by_side

from keepout.conjunction import Conjunction
from keepout.covariance import make_covariance
from keepout.errors import CovarianceError, GeometryError, PcConvergenceError
from keepout.objects import ObjectState
from keepout.orbits import orbital_period
from keepout.pc import (
    AUTO,
    METHODS,
    PcResult,
    alfano,
    api,
    computation,
    compute_pc,
    dilution,
    elrod,
    foster,
    frisbee,
    hall3d,
)
from keepout.pc._encounters import Encounters
from keepout.pc.api import (
    AUTO_MC_SAMPLES,
    AUTO_MC_SEED,
    DURATION_LIMIT,
    SIGMA_EXTENT,
    MethodChoice,
    encounter_duration,
)
from keepout.pc.montecarlo import MC_BATCH, montecarlo
from keepout.pc.result import Method
from keepout.state import StateVector

_period = orbital_period(StateVector(EPOCH, LEO_R, LEO_V, "GCRF"))
assert _period is not None
LEO_PERIOD: float = _period


def test_auto_uses_foster() -> None:
    conjunction = TYPICAL_HEAD_ON
    result = compute_pc(conjunction)
    assert result.method == "foster"
    assert result.value == foster(conjunction.plane, 5.0).value


def test_elrod_can_be_chosen() -> None:
    conjunction = TYPICAL_HEAD_ON
    result = compute_pc(conjunction, "elrod")
    assert result.method == "elrod"
    assert result.value == elrod(conjunction.plane, 5.0).value


def test_options_reach_the_method() -> None:
    conjunction = TYPICAL_HEAD_ON
    result = compute_pc(conjunction, "elrod", order=16)
    assert result.details["order"] == 16
    assert result.value == elrod(conjunction.plane, 5.0, order=16).value


def test_a_looser_tolerance_reaches_foster() -> None:
    loose = compute_pc(TYPICAL_HEAD_ON, "foster", rtol=1e-4)
    default = compute_pc(TYPICAL_HEAD_ON)
    assert loose.details["evaluations"] < default.details["evaluations"]


def test_option_of_another_method_is_rejected() -> None:
    with pytest.raises(
        TypeError, match=r"^foster takes no option \['order'\]; it takes: rtol$"
    ):
        compute_pc(TYPICAL_HEAD_ON, "foster", order=16)  # type: ignore[call-overload]


def _keyword_only(function: Callable[..., object]) -> dict[str, str]:
    return {
        name: str(parameter.annotation)
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }


def _typed_options(overload: Callable[..., object]) -> dict[str, str]:
    """The options an overload types, from its ``**options: Unpack[...]``."""
    for parameter in inspect.signature(overload).parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            unpacked = re.fullmatch(r"Unpack\[(\w+)\]", parameter.annotation)
            assert unpacked is not None
            options = getattr(api, unpacked[1]).__annotations__
            return {key: value.__forward_arg__ for key, value in options.items()}
    return {}


@pytest.mark.parametrize("function", [compute_pc, computation])
def test_overloads_follow_the_methods_options(function: Callable[..., object]) -> None:
    # Each overload types its methods' keyword-only parameters, with their
    # types, so that mypy rejects an option the method does not take.
    functions: dict[str, Callable[..., PcResult]] = {
        "foster": foster,
        "elrod": elrod,
        "frisbee": frisbee,
        "dilution": dilution,
        "alfano": alfano,
        "hall3d": hall3d,
        "montecarlo": montecarlo,
    }
    typed: dict[str, dict[str, str]] = {}
    for overload in typing.get_overloads(function):
        literal = str(inspect.signature(overload).parameters["method"].annotation)
        for name in re.findall(r"'(\w+)'|\"(\w+)\"", literal):
            typed["".join(name)] = _typed_options(overload)
    assert typed == {"auto": {}} | {
        name: _keyword_only(function) for name, function in functions.items()
    }
    assert set(typing.get_args(Method)) == set(functions)
    # The registry holds the same functions and reads their options from them.
    assert {name: spec.function for name, spec in METHODS.items()} == functions
    assert {name: set(spec.options) for name, spec in METHODS.items()} == {
        name: set(_keyword_only(function)) for name, function in functions.items()
    }


def test_a_misspelt_option_fails_the_type_check_and_the_call() -> None:
    # warn_unused_ignores fails mypy if the overloads stop catching it.
    with pytest.raises(TypeError, match=r"foster takes no option \['rtoll'\]"):
        computation("foster", rtoll=1e-9)  # type: ignore[call-overload]


def test_the_registry_says_what_each_method_is() -> None:
    kinds = {name: spec.kind for name, spec in METHODS.items()}
    assert kinds == {
        "foster": "2d",
        "elrod": "2d",
        "frisbee": "maxpc",
        "dilution": "maxpc",
        "alfano": "maxpc",
        "hall3d": "3d",
        "montecarlo": "montecarlo",
    }
    velocity = {name for name, spec in METHODS.items() if spec.uses_velocity}
    assert velocity == {"hall3d", "montecarlo"}


@pytest.mark.parametrize("method", ["frisbee", "dilution", "alfano"])
def test_a_maximum_pc_is_labelled_an_upper_bound(method: Method) -> None:
    conjunction = secondary_unknown() if method == "frisbee" else TYPICAL_HEAD_ON
    assert compute_pc(conjunction, method).kind == "upper_bound"


def test_a_pc_from_both_covariances_is_labelled_an_estimate() -> None:
    assert compute_pc(TYPICAL_HEAD_ON).kind == "estimate"


def test_a_computation_is_compute_pc_with_its_method_and_options() -> None:
    run = computation("elrod", order=16)
    assert (run.method, dict(run.options), run.label) == (
        "elrod",
        {"order": 16},
        "keepout (elrod)",
    )
    assert (
        run(TYPICAL_HEAD_ON).value
        == compute_pc(TYPICAL_HEAD_ON, "elrod", order=16).value
    )
    assert computation().label == "keepout (auto)"
    assert pickle.loads(pickle.dumps(run)).options == {"order": 16}


@pytest.mark.parametrize(
    ("method", "options", "error", "message"),
    [
        ("foster", {"order": 16}, TypeError, r"foster takes no option \['order'\]"),
        ("alfano", {"rtol": 1e-6}, TypeError, "it takes: none$"),
        ("auto", {"rtol": 1e-6}, ValueError, "need an explicit method"),
        ("coppola", {}, ValueError, "unknown Pc method 'coppola'"),
    ],
)
def test_a_computation_checks_its_options_at_once(
    method: str, options: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        computation(method, **options)  # type: ignore[call-overload]


def test_unknown_method_is_rejected_naming_every_method() -> None:
    expected = (
        "unknown Pc method 'coppola'; expected 'auto' or one of ('foster', "
        "'elrod', 'frisbee', 'dilution', 'alfano', 'hall3d', 'montecarlo')"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        compute_pc(TYPICAL_HEAD_ON, "coppola")  # type: ignore[call-overload]


def test_fast_leo_encounter_has_no_warnings() -> None:
    result = compute_pc(TYPICAL_HEAD_ON)
    assert result.diagnostics == ()
    # head_on: v_rel along GCRF -y where the primary's sigma is 1 m, the
    # secondary's 0: 2 (5 * 1 m + 5 m) / (2 * 7546 m/s).
    assert result.details["encounter_duration_s"] == approx(
        2 * (SIGMA_EXTENT * 1.0 + 5.0) / (2 * 7546.0), rel=1e-12
    )


def test_covariance_repairs_come_first_labelled_by_object() -> None:
    # The slow encounter has notes of its own, which the repair must precede.
    base = SLOW_ALONG_TRACK
    asymmetric = known(base.primary.covariance).matrix.copy()
    asymmetric[0, 1] += 0.1 * math.sqrt(asymmetric[0, 0] * asymmetric[1, 1])
    repaired = ObjectState(
        "12345", base.primary.state, make_covariance(asymmetric, "GCRF", EPOCH)
    )
    result = compute_pc(Conjunction(repaired, base.secondary, 5.0))
    own = compute_pc(base).diagnostics
    assert own
    assert result.diagnostics == (
        "primary (12345): symmetrised covariance (max relative asymmetry 0.1)",
        *own,
    )


def along_track_conjunction(duration: float) -> Conjunction:
    """Head-on LEO conjunction whose encounter lasts ``duration`` seconds.

    The relative velocity is along GCRF y, so the along-track sigma sets the
    duration: ``2 (SIGMA_EXTENT sigma + R) / v``, with R = 5 m. Both objects
    have a velocity block (1 mm/s), which ``auto``'s 3-D methods need.
    """
    radius, speed = 5.0, 2 * 7546.0
    sigma = (duration * speed / 2.0 - radius) / SIGMA_EXTENT
    primary = object_state(
        LEO_R, LEO_V, np.diag([100.0, sigma**2, 100.0]), velocity_sigma=1e-3
    )
    secondary = object_state(
        LEO_R + np.array([30.0, 0.0, 0.0]),
        -LEO_V,
        np.zeros((3, 3)),
        velocity_sigma=1e-3,
    )
    return Conjunction(primary, secondary, radius)


@pytest.mark.parametrize(("fraction", "warns"), [(0.99, False), (1.01, True)])
def test_duration_warning_starts_at_the_limit(fraction: float, warns: bool) -> None:
    conjunction = along_track_conjunction(fraction * DURATION_LIMIT * LEO_PERIOD)
    assert encounter_duration(conjunction) == approx(
        fraction * DURATION_LIMIT * LEO_PERIOD, rel=1e-9
    )
    diagnostics = compute_pc(conjunction).diagnostics
    assert (
        any(d.startswith("2-D Pc assumptions questionable") for d in diagnostics)
        is warns
    )


def test_unbound_primary_gets_no_duration_warning() -> None:
    # There is no period to compare the encounter duration with.
    assert compute_pc(unbound_crossing()).diagnostics == ()


def test_auto_keeps_foster_for_a_questionable_unbound_encounter() -> None:
    # Unbound primary, secondary 10 m off closest approach along the relative
    # velocity (z): a 2-D assumption is questionable, but Hall needs elliptic
    # orbits, so Foster stays, with the note and without a 2-D comparison.
    result = compute_pc(unbound_crossing(z_offset=10.0))
    assert result.method == "foster"
    (message,) = result.diagnostics
    assert message.startswith("states are 10 m from closest approach")
    assert "pc_2d" not in result.details


def test_states_away_from_tca_are_reported() -> None:
    # Secondary 10 m along the relative velocity from closest approach.
    primary = object_state(LEO_R, LEO_V, 100.0 * np.eye(3), velocity_sigma=1e-3)
    secondary = object_state(
        LEO_R + np.array([30.0, -10.0, 0.0]), -LEO_V, np.eye(3), velocity_sigma=1e-3
    )
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
    assert result.details["encounter_duration_s"] == approx(
        2 * (SIGMA_EXTENT * math.sqrt(2.0) + 5.0) / 20.0, rel=1e-9
    )


SLOW_ALONG_TRACK = along_track_conjunction(1.5 * DURATION_LIMIT * LEO_PERIOD)
"""An encounter long enough for the 2-D assumptions to be questionable."""


def position_only_conjunction(*, both: bool) -> Conjunction:
    """:data:`SLOW_ALONG_TRACK` with the primary's covariance known only in
    position, and with ``both`` the secondary's too."""
    moving = SLOW_ALONG_TRACK
    primary = object_state(LEO_R, LEO_V, known(moving.primary.covariance).position)
    secondary = (
        object_state(LEO_R + np.array([30.0, 0.0, 0.0]), -LEO_V, np.eye(3))
        if both
        else moving.secondary
    )
    return Conjunction(primary, secondary, 5.0)


POSITION_ONLY_CASES = pytest.mark.parametrize(
    ("both", "roles", "labels"),
    [
        (False, ("primary",), "covariance of primary (TEST) is"),
        (
            True,
            ("primary", "secondary"),
            "covariances of primary (TEST) and secondary (TEST) are",
        ),
    ],
)


@POSITION_ONLY_CASES
def test_auto_keeps_the_2d_pc_for_covariances_known_only_in_position(
    both: bool, roles: tuple[str, ...], labels: str
) -> None:
    # A questionable 2-D Pc would take auto to Hall, but with a zero
    # velocity block Hall would treat the velocity as exact.
    conjunction = position_only_conjunction(both=both)
    assert conjunction.position_only_covariance == roles
    result = compute_pc(conjunction)
    assert result.method == "foster"
    assert result.value == foster(conjunction.plane, 5.0).value
    assert result.diagnostics[0].startswith("2-D Pc assumptions questionable")
    assert result.diagnostics[1] == (
        f"the {labels} known only in position (zero velocity block): the "
        "3-D methods need the velocity uncertainty, so the 2-D Pc is used"
    )
    assert "pc_2d" not in result.details


@POSITION_ONLY_CASES
def test_an_explicit_3d_method_notes_a_covariance_known_only_in_position(
    both: bool, roles: tuple[str, ...], labels: str
) -> None:
    result = compute_pc(position_only_conjunction(both=both), "hall3d")
    assert result.method == "hall3d"
    assert (
        f"the {labels} known only in position (zero velocity block): hall3d "
        "takes its velocity as exact"
    ) in result.diagnostics


def test_a_2d_method_needs_no_velocity_note() -> None:
    result = compute_pc(position_only_conjunction(both=True), "foster")
    assert not any("velocity as exact" in note for note in result.diagnostics)


def test_a_sound_2d_pc_needs_no_velocity_block() -> None:
    assert TYPICAL_HEAD_ON.position_only_covariance == ("primary", "secondary")
    result = compute_pc(TYPICAL_HEAD_ON)
    assert result.method == "foster"
    assert result.diagnostics == ()


FAR_HEAD_ON = head_on(300.0, 100.0, 200.0, 5.0)
"""A head-on encounter 300 m off, with known covariances."""


def secondary_unknown() -> Conjunction:
    """:data:`FAR_HEAD_ON` with the secondary's covariance unknown."""
    base = FAR_HEAD_ON
    return Conjunction(base.primary, without_covariance(base.secondary), 5.0)


def both_unknown(base: Conjunction) -> Conjunction:
    """``base`` with both covariances unknown."""
    return Conjunction(
        without_covariance(base.primary),
        without_covariance(base.secondary),
        base.combined_hard_body_radius,
    )


def test_auto_uses_frisbee_when_one_covariance_is_unknown() -> None:
    conjunction = secondary_unknown()
    result = compute_pc(conjunction)
    assert result.method == "frisbee"
    assert result.diagnostics[0] == "secondary (SEC): covariance unknown"
    assert "upper bound" in result.diagnostics[1]


def test_auto_uses_alfano_when_both_covariances_are_unknown() -> None:
    conjunction = both_unknown(FAR_HEAD_ON)
    result = compute_pc(conjunction)
    assert result.method == "alfano"
    assert result.value == approx(
        5.0 / 300.0 * math.sqrt(2.0 / (math.pi * math.e)), rel=1e-3
    )


def test_frisbee_can_be_chosen() -> None:
    conjunction = secondary_unknown()
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
    slow = SLOW_ALONG_TRACK
    conjunction = Conjunction(slow.primary, without_covariance(slow.secondary), 5.0)
    result = compute_pc(conjunction, method)
    assert result.method == "frisbee"
    assert result.diagnostics[1].startswith(
        "2-D Pc assumptions questionable: the encounter lasts at least about"
    )
    assert "encounter_duration_s" not in result.details


@pytest.mark.parametrize("method", ["auto", "alfano"])
def test_maximum_pc_keeps_the_distance_along_velocity_note(
    method: MethodChoice,
) -> None:
    # The offset does not depend on the covariances.
    conjunction = both_unknown(shifted_back(FAR_HEAD_ON, 1.0))
    result = compute_pc(conjunction, method)
    assert result.method == "alfano"
    assert any(
        d.startswith("states are 1.51e+04 m from closest") for d in result.diagnostics
    )
    assert "encounter_duration_s" not in result.details


def test_alfano_can_be_chosen() -> None:
    conjunction = both_unknown(FAR_HEAD_ON)
    result = compute_pc(conjunction, "alfano")
    assert result.method == "alfano"
    assert result.value == alfano(conjunction.plane, 5.0).value
    assert "encounter_duration_s" not in result.details


def test_options_reach_a_three_d_method() -> None:
    conjunction = TYPICAL_HEAD_ON
    result = compute_pc(conjunction, "montecarlo", samples=2_000, seed=1)
    assert result.method == "montecarlo"
    assert result.details["samples"] == 2_000
    assert result.value == montecarlo(conjunction, samples=2_000, seed=1).value


@pytest.mark.parametrize(
    "method", ["foster", "elrod", "dilution", "hall3d", "montecarlo"]
)
def test_methods_needing_both_covariances_refuse_an_unknown_one(
    method: MethodChoice,
) -> None:
    conjunction = secondary_unknown()
    with pytest.raises(CovarianceError, match="needs both covariances"):
        compute_pc(conjunction, method)


def test_auto_uses_hall_for_a_slow_encounter_and_keeps_the_2d_pc() -> None:
    conjunction = SLOW_ALONG_TRACK
    result = compute_pc(conjunction)
    assert result.method == "hall3d"
    assert result.details["pc_2d"] == foster(conjunction.plane, 5.0).value
    assert result.diagnostics[0].startswith("2-D Pc assumptions questionable")


@pytest.mark.parametrize("method", ["auto", "hall3d"])
def test_a_long_encounter_is_described_once(method: MethodChoice) -> None:
    # The duration is in the applicability warning; Hall only adds what it
    # means for Nc.
    diagnostics = compute_pc(SLOW_ALONG_TRACK, method).diagnostics
    assert [d for d in diagnostics if "orbital period" in d] == [diagnostics[0]]
    assert (
        "long encounter: Nc counts repeated entries into the hard-body sphere, so "
        "it is an upper bound on Pc"
    ) in diagnostics


def _scan_spy(monkeypatch: pytest.MonkeyPatch) -> list[Encounters]:
    """Record every encounter scan made through :mod:`keepout.pc.api`."""
    scans: list[Encounters] = []
    real = Encounters.locate

    def spy(conjunction: Conjunction, window: tuple[float, float] | None) -> Encounters:
        scans.append(real(conjunction, window))
        return scans[-1]

    monkeypatch.setattr(Encounters, "locate", spy)
    return scans


def test_auto_falls_back_to_monte_carlo_on_hall_s_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: Conjunction, encounters: Encounters, **options: object) -> PcResult:
        raise PcConvergenceError(
            "Hall 3D-Nc: the peak-overlap iteration did not converge"
        )

    calls: list[tuple[Encounters, dict[str, object]]] = []

    def run(_: Conjunction, encounters: Encounters, **options: object) -> PcResult:
        calls.append((encounters, options))
        return PcResult(0.25, "montecarlo", (), {})

    scans = _scan_spy(monkeypatch)
    auto_hall_integration(monkeypatch, fail)
    auto_monte_carlo(monkeypatch, run)
    result = compute_pc(SLOW_ALONG_TRACK)
    assert result.method == "montecarlo"
    assert result.diagnostics[-1].endswith(
        "; Monte Carlo used instead (seed 0, at most 1,000,000 samples)"
    )
    # Monte Carlo reuses Hall's scan; a fixed seed and a bounded budget keep
    # ``auto`` reproducible and bounded.
    (scan,) = scans
    assert calls == [(scan, {"max_samples": AUTO_MC_SAMPLES, "seed": AUTO_MC_SEED})]


def test_the_resolved_fallback_reproduces_autos_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlap_failing_while_integrating(monkeypatch)
    auto_monte_carlo_budget(monkeypatch, MC_BATCH)
    result = AUTO(SLOW_ALONG_TRACK)
    resolved = AUTO.resolved(result)
    assert resolved.method == "montecarlo"
    assert resolved(SLOW_ALONG_TRACK).value == result.value


def test_auto_resolves_to_the_method_it_chose_with_its_defaults() -> None:
    result = AUTO(TYPICAL_HEAD_ON)
    assert result.method == "foster"
    resolved = AUTO.resolved(result)
    assert (resolved.method, dict(resolved.options)) == ("foster", {})


def test_an_explicit_computation_resolves_to_itself() -> None:
    run = computation("elrod", order=16)
    assert run.resolved(run(TYPICAL_HEAD_ON)) is run
    with pytest.raises(ValueError, match="a foster result is not one of elrod's"):
        run.resolved(compute_pc(TYPICAL_HEAD_ON, "foster"))


def test_fallback_matches_an_explicit_monte_carlo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The peak-overlap iteration fails at Hall's quadrature nodes only, after
    # the scan: Hall refuses the result, and Monte Carlo, which reuses the
    # scan, must still run as if alone (the failures it checks are the scan's).
    overlap_failing_while_integrating(monkeypatch)
    auto_monte_carlo_budget(monkeypatch, MC_BATCH)
    conjunction = SLOW_ALONG_TRACK
    result = compute_pc(conjunction)
    assert result.method == "montecarlo"
    assert any(
        "peak-overlap iteration did not converge" in note
        and note.endswith("Monte Carlo used instead (seed 0, at most 50,000 samples)")
        for note in result.diagnostics
    )
    alone = montecarlo(conjunction, max_samples=MC_BATCH, seed=AUTO_MC_SEED)
    assert result.value == alone.value
    # compute_pc adds its own details to the method's.
    assert result.details == {
        **alone.details,
        "encounter_duration_s": encounter_duration(conjunction),
        "pc_2d": foster(conjunction.plane, 5.0).value,
    }


def test_auto_uses_hall_at_zero_relative_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Hall's accuracy at zero relative velocity is checked on Alfano case 12
    # (validation); here only the choice is.
    def run(_: Conjunction, encounters: Encounters) -> PcResult:
        return PcResult(0.5, "hall3d", (), {})

    auto_hall_integration(monkeypatch, run)
    result = compute_pc(side_by_side())
    assert result.method == "hall3d"
    assert result.diagnostics == (
        "relative velocity is zero: 2-D Pc is undefined",
        (
            "the covariances of primary (PRI) and secondary (SEC) are known only "
            "in position (zero velocity block): hall3d takes its velocity as exact"
        ),
    )
    assert "encounter_duration_s" not in result.details


@pytest.mark.parametrize(
    ("unknown", "message"),
    [
        (("secondary",), r"the covariance of secondary \(SEC\) is unknown$"),
        (
            ("primary", "secondary"),
            r"the covariances of primary \(PRI\) and secondary \(SEC\) are unknown$",
        ),
    ],
)
def test_auto_refuses_zero_relative_velocity_with_an_unknown_covariance(
    unknown: tuple[str, ...], message: str
) -> None:
    # Max-Pc needs the encounter plane and the 3-D methods both covariances.
    base = side_by_side()
    primary, secondary = (
        without_covariance(item) if role in unknown else item
        for role, item in (("primary", base.primary), ("secondary", base.secondary))
    )
    conjunction = Conjunction(primary, secondary, base.combined_hard_body_radius)
    with pytest.raises(GeometryError, match=f"velocity is zero .* {message}"):
        compute_pc(conjunction)


def test_auto_refuses_zero_relative_velocity_with_an_unbound_orbit() -> None:
    # The scan's error names the object; no method is named, as none ran.
    with pytest.raises(
        GeometryError,
        match=(
            r"^no Pc method applies: the relative velocity is zero \(no encounter "
            r"plane\) and the 3-D methods cannot use the orbit of primary \(PRI\): "
            r".* the state is unbound$"
        ),
    ):
        compute_pc(side_by_side(1.5 * LEO_V))


def test_options_need_an_explicit_method() -> None:
    with pytest.raises(ValueError, match="need an explicit method"):
        compute_pc(TYPICAL_HEAD_ON, rtol=1e-6)  # type: ignore[call-overload]
