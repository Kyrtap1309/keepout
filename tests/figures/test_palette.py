import itertools
import math
import re
from collections.abc import Mapping
from enum import Enum

import numpy as np
import pytest

from keepout._types import FloatArray
from keepout.figures import (
    ACTION_COLOURS,
    ACTION_LABELS,
    RISK_COLOURS,
    RISK_LABELS,
    SEQUENTIAL,
    SERIES,
)
from keepout.risk import Action, RiskLevel

HEX = re.compile(r"^#[0-9a-f]{6}$")

# Machado, Oliveira & Fernandes (2009), severity 1.0, on linear RGB.
PROTANOPIA = np.array(
    [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]
)
DEUTERANOPIA = np.array(
    [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]
)
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
D65 = np.array([0.95047, 1.0, 1.08883])
SMALLEST_DIFFERENCE = 20.0


def linear_rgb(colour: str) -> FloatArray:
    """The linear RGB of a ``#rrggbb`` colour, in [0, 1]."""
    srgb = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def lab(rgb: FloatArray) -> tuple[float, float, float]:
    """CIELAB (D65) of a linear RGB colour, clipped to the gamut."""
    xyz = SRGB_TO_XYZ @ np.clip(rgb, 0.0, 1.0) / D65
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3.0 * delta**2) + 4.0 / 29.0)
    return (
        float(116.0 * f[1] - 16.0),
        float(500.0 * (f[0] - f[1])),
        float(200.0 * (f[1] - f[2])),
    )


def _mean_hue(h1: float, h2: float) -> float:
    if abs(h1 - h2) <= 180.0:
        return (h1 + h2) / 2.0
    return (h1 + h2 + 360.0) / 2.0 if h1 + h2 < 360.0 else (h1 + h2 - 360.0) / 2.0


def ciede2000(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    """The CIEDE2000 colour difference (Sharma, Wu & Dalal 2005)."""
    l1, a1, b1 = first
    l2, a2, b2 = second
    mean_c = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2.0
    g = 0.5 * (1.0 - math.sqrt(mean_c**7 / (mean_c**7 + 25.0**7)))
    a1, a2 = (1.0 + g) * a1, (1.0 + g) * a2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    h1 = math.degrees(math.atan2(b1, a1)) % 360.0
    h2 = math.degrees(math.atan2(b2, a2)) % 360.0
    chromatic = c1 * c2 != 0.0
    dh = (h2 - h1 + 180.0) % 360.0 - 180.0 if chromatic else 0.0
    d_l, d_c = l2 - l1, c2 - c1
    d_h = 2.0 * math.sqrt(c1 * c2) * math.sin(math.radians(dh / 2.0))
    mean_l, mean_cp = (l1 + l2) / 2.0, (c1 + c2) / 2.0
    mean_h = _mean_hue(h1, h2) if chromatic else h1 + h2
    t = (
        1.0
        - 0.17 * math.cos(math.radians(mean_h - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * mean_h))
        + 0.32 * math.cos(math.radians(3.0 * mean_h + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * mean_h - 63.0))
    )
    rotation = 30.0 * math.exp(-(((mean_h - 275.0) / 25.0) ** 2))
    r_c = 2.0 * math.sqrt(mean_cp**7 / (mean_cp**7 + 25.0**7))
    s_l = 1.0 + 0.015 * (mean_l - 50.0) ** 2 / math.sqrt(20.0 + (mean_l - 50.0) ** 2)
    s_c = 1.0 + 0.045 * mean_cp
    s_h = 1.0 + 0.015 * mean_cp * t
    r_t = -math.sin(math.radians(2.0 * rotation)) * r_c
    return math.sqrt(
        (d_l / s_l) ** 2
        + (d_c / s_c) ** 2
        + (d_h / s_h) ** 2
        + r_t * (d_c / s_c) * (d_h / s_h)
    )


def test_ciede2000_matches_sharmas_reference_pair() -> None:
    # Pair 1 of Sharma, Wu & Dalal's test data.
    assert ciede2000((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485)) == (
        pytest.approx(2.0425, abs=1e-4)
    )


@pytest.mark.parametrize(
    ("mapping", "members"),
    [
        (RISK_COLOURS, RiskLevel),
        (RISK_LABELS, RiskLevel),
        (ACTION_COLOURS, Action),
        (ACTION_LABELS, Action),
    ],
)
def test_palette_is_distinct_and_hex(
    mapping: Mapping[Enum, str], members: type[Enum]
) -> None:
    assert set(mapping) == set(members)
    assert len(set(mapping.values())) == len(mapping)
    if mapping in (RISK_COLOURS, ACTION_COLOURS):
        assert all(HEX.match(colour) for colour in mapping.values())


def test_sequences_are_distinct_and_hex() -> None:
    for colours in (SERIES, SEQUENTIAL):
        assert len(set(colours)) == len(colours)
        assert all(HEX.match(colour) for colour in colours)


def test_risk_labels_are_the_level_names() -> None:
    assert [RISK_LABELS[level] for level in RiskLevel] == ["green", "yellow", "red"]


@pytest.mark.parametrize("simulation", [DEUTERANOPIA, PROTANOPIA])
def test_risk_colours_are_colour_blind_safe(simulation: FloatArray) -> None:
    seen = [lab(simulation @ linear_rgb(RISK_COLOURS[level])) for level in RiskLevel]
    for first, second in itertools.combinations(seen, 2):
        assert ciede2000(first, second) > SMALLEST_DIFFERENCE
