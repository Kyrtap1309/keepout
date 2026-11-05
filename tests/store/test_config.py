import re
from pathlib import Path

import pytest

from keepout._defaults import TARGET_PC
from keepout.errors import StoreError
from keepout.pc import AUTO, computation
from keepout.risk import DEFAULT_THRESHOLDS, Thresholds
from keepout.store.codec import result_key
from keepout.store.config import Settings, load_settings


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "keepout.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_settings() -> None:
    settings = Settings()
    assert settings.compute == AUTO
    assert settings.policy == "repair"
    assert settings.combined_hard_body_radius is None
    assert settings.thresholds == DEFAULT_THRESHOLDS
    assert settings.target_pc == TARGET_PC


def test_the_result_key_of_settings() -> None:
    settings = Settings(
        compute=computation("foster"), policy="raise", combined_hard_body_radius=5.0
    )
    assert settings.result_key == result_key(
        compute=computation("foster"), policy="raise", combined_hard_body_radius=5.0
    )
    # The thresholds and the target are applied on read: same cache.
    assert (
        Settings(thresholds=Thresholds(1e-6, 1e-3), target_pc=1e-6).result_key
        == Settings().result_key
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"policy": "ignore"}, "policy"),
        ({"combined_hard_body_radius": 0.0}, "combined_hard_body_radius"),
        ({"target_pc": 1.0}, "target_pc"),
        ({"target_pc": 0.0}, "target_pc"),
    ],
)
def test_invalid_settings(changes: dict[str, object], message: str) -> None:
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
    assert (
        settings.result_key
        == Settings(
            compute=computation("foster"),
            policy="raise",
            combined_hard_body_radius=20.0,
        ).result_key
    )


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
        StoreError, match=rf"{re.escape(str(path))}: unknown key\(s\) {key}"
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
    with pytest.raises(StoreError, match=rf"{re.escape(str(path))}: {key}"):
        load_settings(path)


def test_settings_that_are_not_toml(tmp_path: Path) -> None:
    path = _write(tmp_path, "[pc\n")
    with pytest.raises(StoreError, match=rf"{re.escape(str(path))}: not TOML"):
        load_settings(path)
    path.write_bytes(b"\xff")
    with pytest.raises(StoreError, match=rf"{re.escape(str(path))}: not UTF-8"):
        load_settings(path)
