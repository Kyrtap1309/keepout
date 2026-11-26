from pathlib import Path

import operator_example as example


def test_paths_are_shown_from_the_repository() -> None:
    assert example.shown(example.FLEET) == "examples/synthetic-operator/fleet.toml"
    assert example.shown(Path("/elsewhere/fleet.toml")) == "/elsewhere/fleet.toml"


def test_rounded_shortens_real_values_only() -> None:
    text = "\n".join(
        [
            "CCSDS_CDM_VERS                 = 2.0",
            "COMMENT Sigmas: R 0.204 km, T 19.579123456789 km",
            "TCA                            = 2026-10-12T11:28:28.709977",
            "MESSAGE_ID                     = 38771_35196_20261012T112828.710Z",
            "OBJECT_DESIGNATOR              = 38771",
            "MISS_DISTANCE                  = 212.3720068512821 [m]",
            "Z_DOT                          = -0.35235483308181775 [km/s]",
            "CR_R                           = 41616.00000000085 [m**2]",
            "CT_R                           = 6.058122821e-09 [m**2]",
            "CN_T                           = -1.8e-07 [m**2]",
            "CT_T                           = 1.2345678901234e+20 [m**2]",
            "CD_AREA_OVER_MASS              = 0.012345678901234",
            "",
        ]
    )
    assert example.rounded(text, 10).split("\n") == [
        "CCSDS_CDM_VERS                 = 2.0",
        "COMMENT Sigmas: R 0.204 km, T 19.579123456789 km",
        "TCA                            = 2026-10-12T11:28:28.709977",
        "MESSAGE_ID                     = 38771_35196_20261012T112828.710Z",
        "OBJECT_DESIGNATOR              = 38771",
        "MISS_DISTANCE                  = 212.3720069 [m]",
        "Z_DOT                          = -0.3523548331 [km/s]",
        "CR_R                           = 41616.0 [m**2]",
        "CT_R                           = 0.0 [m**2]",
        "CN_T                           = 0.0 [m**2]",
        "CT_T                           = 1.23456789e+20 [m**2]",
        "CD_AREA_OVER_MASS              = 0.0123456789",
        "",
    ]


def test_rounded_takes_the_digits_and_the_floor() -> None:
    assert example.rounded("X = 1.23456 [km]", digits=3, floor=0.0) == "X = 1.23 [km]"
