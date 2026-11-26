import re
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from keepout._defaults import TARGET_PC
from keepout._methods import METHOD_CHOICES
from keepout.covariance import POLICIES
from keepout.errors import SettingsError
from keepout.pc import AUTO, PcComputation
from keepout.risk import DEFAULT_THRESHOLDS, Thresholds
from keepout.settings import (
    SETTINGS_KEYS,
    Settings,
    SettingsData,
    load_settings,
    settings_from_data,
    settings_to_data,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "keepout.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_settings_keys_are_those_of_a_full_settings_file() -> None:
    # Every table and key settings_to_data gives (hbr_m once a radius is set).
    full = settings_to_data(Settings(combined_hard_body_radius=20.0))
    assert list(SETTINGS_KEYS) == list(SettingsData.__annotations__)
    assert {
        "pc": set(full["pc"]),
        "risk": set(full["risk"]),
        "maneuver": set(full["maneuver"]),
    } == {name: set(keys) for name, keys in SETTINGS_KEYS.items()}


def test_settings_default_to_the_librarys() -> None:
    settings = Settings()
    assert settings.compute == AUTO
    assert settings.policy == "repair"
    assert settings.combined_hard_body_radius is None
    assert settings.thresholds == DEFAULT_THRESHOLDS
    assert settings.target_pc == TARGET_PC


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"policy": "ignore"}, "policy"),
        ({"combined_hard_body_radius": 0.0}, "combined_hard_body_radius"),
        ({"target_pc": 1.0}, "target_pc"),
        ({"target_pc": 0.0}, "target_pc"),
    ],
)
def test_invalid_settings_are_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Settings(**changes)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_settings_from_toml(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
[pc]
method = "foster"
policy = "raise"
hbr_m = 20

[risk]
green_below = 1e-6
red_from = 1e-3

[maneuver]
target_pc = 1e-6
""",
    )
    settings = load_settings(path)
    # PcComputation compares by identity: compare what it computes.
    assert (settings.compute.method, dict(settings.compute.options)) == ("foster", {})
    assert (settings.policy, settings.combined_hard_body_radius) == ("raise", 20.0)
    assert settings.thresholds == Thresholds(1e-6, 1e-3)
    assert settings.target_pc == 1e-6


def test_missing_keys_take_the_defaults(tmp_path: Path) -> None:
    assert load_settings(_write(tmp_path, "")) == Settings()
    path = _write(tmp_path, "[risk]\nred_from = 1e-3\n")
    assert load_settings(path).thresholds == Thresholds(
        DEFAULT_THRESHOLDS.green_below, 1e-3
    )


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[pc]\nmethods = 'foster'\n", "pc.methods"),
        ("[risk]\nbasis = 'estimate'\n", "risk.basis"),
        ("[maneuvre]\ntarget_pc = 1e-6\n", "maneuvre"),
        ("target_pc = 1e-6\n", "target_pc"),
    ],
)
def test_unknown_settings_key_is_named(tmp_path: Path, text: str, key: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(
        SettingsError, match=rf"{re.escape(str(path))}: unknown key\(s\) {key}"
    ):
        load_settings(path)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[pc]\nmethod = 'magic'\n", "pc.method"),
        ("[pc]\nmethod = 1\n", "pc.method"),
        ("[pc]\npolicy = 'ignore'\n", "pc.policy"),
        ("[pc]\nhbr_m = -1.0\n", "pc.hbr_m"),
        ("[pc]\nhbr_m = '20'\n", "pc.hbr_m"),
        ("[pc]\nhbr_m = true\n", "pc.hbr_m"),
        ("[risk]\ngreen_below = 1e-3\n", "risk"),
        ("[risk]\nred_from = 'high'\n", "risk.red_from"),
        ("[maneuver]\ntarget_pc = 2.0\n", "maneuver.target_pc"),
        ("pc = 1\n", "pc"),
    ],
)
def test_a_bad_settings_value_is_named(tmp_path: Path, text: str, key: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(SettingsError, match=rf"{re.escape(str(path))}: {key}"):
        load_settings(path)


def test_settings_that_are_not_toml(tmp_path: Path) -> None:
    path = _write(tmp_path, "[pc\n")
    with pytest.raises(SettingsError, match=rf"{re.escape(str(path))}: not TOML"):
        load_settings(path)
    path.write_bytes(b"\xff")
    with pytest.raises(SettingsError, match=rf"{re.escape(str(path))}: not UTF-8"):
        load_settings(path)


def _assert_same_settings(actual: Settings, expected: Settings) -> None:
    """Settings equal, ``compute`` field by field: :class:`PcComputation`
    compares by identity."""
    assert (
        actual.compute.method,
        dict(actual.compute.options),
        actual.compute.fallback,
    ) == (
        expected.compute.method,
        dict(expected.compute.options),
        expected.compute.fallback,
    )
    assert replace(actual, compute=expected.compute) == expected


@pytest.mark.parametrize(
    "settings",
    [
        Settings(),
        Settings(
            compute=PcComputation("foster"),
            policy="raise",
            combined_hard_body_radius=20.0,
            thresholds=replace(DEFAULT_THRESHOLDS, green_below=1e-6, red_from=1e-4),
            target_pc=1e-6,
        ),
    ],
)
def test_settings_round_trip(settings: Settings) -> None:
    _assert_same_settings(settings_from_data(settings_to_data(settings)), settings)


def test_settings_data_has_every_key() -> None:
    data = settings_to_data(Settings(combined_hard_body_radius=20.0))
    assert set(data) == {"pc", "risk", "maneuver"}
    assert set(data["pc"]) == {"method", "policy", "hbr_m"}
    assert set(data["risk"]) == {"green_below", "red_from"}
    assert set(data["maneuver"]) == {"target_pc"}
    assert "hbr_m" not in settings_to_data(Settings())["pc"]


_PROBABILITIES = st.floats(min_value=1e-300, max_value=1.0, exclude_max=True)


@st.composite
def settings_files_hold(draw: st.DrawFn) -> Settings:
    """Settings a settings file can hold: any but options, a fallback or
    another basis."""
    green, red = sorted(draw(st.lists(_PROBABILITIES, min_size=2, max_size=2)))
    if green == red:
        red = 1.0
    return Settings(
        compute=PcComputation(draw(st.sampled_from(METHOD_CHOICES))),
        policy=draw(st.sampled_from(POLICIES)),
        combined_hard_body_radius=draw(
            st.none() | st.floats(min_value=1e-3, max_value=1e4)
        ),
        thresholds=Thresholds(green_below=green, red_from=red),
        target_pc=draw(_PROBABILITIES),
    )


@given(settings_files_hold())
def test_any_settings_round_trip(settings: Settings) -> None:
    _assert_same_settings(settings_from_data(settings_to_data(settings)), settings)


@pytest.mark.parametrize(
    ("settings", "field"),
    [
        (
            Settings(compute=PcComputation("montecarlo", options={"seed": 1})),
            "compute.options",
        ),
        (
            Settings(
                compute=PcComputation("hall3d", fallback=PcComputation("montecarlo"))
            ),
            "compute.fallback",
        ),
        (
            Settings(thresholds=replace(DEFAULT_THRESHOLDS, basis="estimate")),
            "thresholds.basis",
        ),
    ],
)
def test_settings_a_file_cannot_hold_are_refused(
    settings: Settings, field: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(field)):
        settings_to_data(settings)
