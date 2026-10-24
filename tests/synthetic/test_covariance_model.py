"""The synthetic covariance of GP element sets: regimes, parameters and growth
with the element set's age."""

import dataclasses
import math
import pickle

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from hypothesis import given
from hypothesis import strategies as st
from scipy.optimize import brentq
from screening_data import fixture_by_number

from keepout.covariance import problems
from keepout.errors import UnsupportedElements
from keepout.omm import Omm
from keepout.propagation import satrec_of
from keepout.synthetic import (
    OrbitRegime,
    RegimeParameters,
    RtnSigmas,
    SyntheticCovarianceModel,
    orbit_regime,
)
from keepout.synthetic.covariance_model import (
    FLOHRER_SIGMAS,
    GEO_HIGHEST_APOGEE,
    GEO_LOWEST_PERIGEE,
    LEO_HIGHEST_APOGEE,
    LEVIT_MARSHALL_RATE,
    PUBLISHED_PARAMETERS,
    SAFETY_FACTOR,
)

BY_NUMBER = fixture_by_number()
"""The 222 objects of the frozen public catalog's fixture."""

EXAMPLES = (900, 25933, 26113, 14129, 25924)
"""One object of each regime: LEO, MEO, GTO, HEO, GEO."""

MODEL = SyntheticCovarianceModel()

DAYS = 30.0 * 86400.0
elements = st.sampled_from([BY_NUMBER[number] for number in EXAMPLES])
ages = st.floats(min_value=-DAYS, max_value=DAYS)


def at(omm: Omm, age: float) -> Time:
    """The time ``age`` seconds after the element epoch."""
    time: Time = omm.epoch + age * u.s
    return time


def test_the_published_values() -> None:
    assert FLOHRER_SIGMAS == {
        OrbitRegime.LEO: (102.0, 471.0, 126.0),
        OrbitRegime.MEO: (73.0, 131.0, 54.0),
        OrbitRegime.GTO: (1960.0, 3897.0, 1808.0),
        OrbitRegime.HEO: (824.0, 1367.0, 1059.0),
        OrbitRegime.GEO: (359.0, 432.0, 86.0),
    }
    assert pytest.approx(1500.0, rel=1e-15) == LEVIT_MARSHALL_RATE * 86400.0
    assert SAFETY_FACTOR == 2.0
    for regime in OrbitRegime:
        radial, transverse, normal = FLOHRER_SIGMAS[regime]
        expected = RegimeParameters(radial, transverse, normal, LEVIT_MARSHALL_RATE)
        assert PUBLISHED_PARAMETERS[regime] == expected
        assert MODEL.parameters(regime) == expected
    assert MODEL.safety_factor == SAFETY_FACTOR


@pytest.mark.parametrize(
    ("number", "regime"),
    [
        (900, OrbitRegime.LEO),  # CALSPHERE 1, 954 x 991 km
        (31631, OrbitRegime.LEO),  # FENGYUN 1C DEB, 450 x 490 km
        (25933, OrbitRegime.MEO),  # NAVSTAR 46 (GPS)
        (1361, OrbitRegime.MEO),  # LCS 1, 2781 x 2801 km
        (2866, OrbitRegime.MEO),  # LES-5, apogee 33 611 km
        (26113, OrbitRegime.GTO),  # IMAGE, 1481 x 45 380 km
        (47719, OrbitRegime.GTO),  # ARKTIKA-M 1 (Molniya-like), 726 x 39 624 km
        (14129, OrbitRegime.HEO),  # AO-10, 4125 x 35 328 km
        (25867, OrbitRegime.HEO),  # CXO, apogee 136 315 km
        (270214, OrbitRegime.HEO),  # 431 x 2866 km: apogee above LEO's
        (42965, OrbitRegime.HEO),  # QZS-4, 32 625 x 38 951 km
        (25924, OrbitRegime.GEO),  # ABS-6, geostationary
        (44204, OrbitRegime.GEO),  # BEIDOU-3 IGSO-1: by altitude only
    ],
)
def test_regimes_of_real_orbits(number: int, regime: OrbitRegime) -> None:
    assert orbit_regime(BY_NUMBER[number]) is regime


def sgp4_altitudes(omm: Omm) -> tuple[float, float]:
    """SGP4's perigee and apogee altitudes [m] of an element set."""
    satrec = satrec_of(omm)
    radius = satrec.radiusearthkm * 1e3
    return float(satrec.altp * radius), float(satrec.alta * radius)


def with_altitudes(perigee: float, apogee: float) -> Omm:
    """CALSPHERE 1 moved to SGP4 perigee and apogee altitudes [m].

    The eccentricity fixes their ratio; the mean motion (found by root
    search) the perigee.
    """
    base = BY_NUMBER[900]
    radius = satrec_of(base).radiusearthkm * 1e3
    eccentricity = (apogee - perigee) / (apogee + perigee + 2.0 * radius)
    shaped = dataclasses.replace(base, eccentricity=eccentricity)

    def miss(mean_motion: float) -> float:
        moved = dataclasses.replace(shaped, mean_motion=mean_motion)
        return sgp4_altitudes(moved)[0] - perigee

    found = brentq(miss, 1e-5, 1.5e-3, xtol=1e-18)
    moved = dataclasses.replace(shaped, mean_motion=found)
    assert sgp4_altitudes(moved) == pytest.approx((perigee, apogee), abs=0.1)
    return moved


@pytest.mark.parametrize(
    ("perigee", "apogee", "regime"),
    [
        (500e3, LEO_HIGHEST_APOGEE - 1.0, OrbitRegime.LEO),
        (500e3, LEO_HIGHEST_APOGEE + 1.0, OrbitRegime.HEO),
        (LEO_HIGHEST_APOGEE + 1.0, 20_000e3, OrbitRegime.MEO),
        (LEO_HIGHEST_APOGEE - 1.0, 20_000e3, OrbitRegime.HEO),
        (20_000e3, GEO_LOWEST_PERIGEE - 1.0, OrbitRegime.MEO),
        (20_000e3, GEO_LOWEST_PERIGEE + 1.0, OrbitRegime.HEO),
        (GEO_LOWEST_PERIGEE + 1.0, 35_786e3, OrbitRegime.GEO),
        (GEO_LOWEST_PERIGEE - 1.0, 35_786e3, OrbitRegime.HEO),
        (35_000e3, GEO_HIGHEST_APOGEE - 1.0, OrbitRegime.GEO),
        (35_000e3, GEO_HIGHEST_APOGEE + 1.0, OrbitRegime.HEO),
        (500e3, GEO_LOWEST_PERIGEE + 1.0, OrbitRegime.GTO),
        (500e3, GEO_LOWEST_PERIGEE - 1.0, OrbitRegime.HEO),
        (LEO_HIGHEST_APOGEE - 1.0, 40_000e3, OrbitRegime.GTO),
        (LEO_HIGHEST_APOGEE + 1.0, 40_000e3, OrbitRegime.HEO),
    ],
)
def test_regimes_are_bounded_by_the_sgp4_altitudes(
    perigee: float, apogee: float, regime: OrbitRegime
) -> None:
    assert orbit_regime(with_altitudes(perigee, apogee)) is regime


def test_regimes_need_an_sgp4_element_set() -> None:
    beyond = dataclasses.replace(BY_NUMBER[900], catalog_number=340_000)
    with pytest.raises(UnsupportedElements, match="339999"):
        orbit_regime(beyond)
    with pytest.raises(UnsupportedElements, match="339999"):
        MODEL.covariance(beyond, beyond.epoch)


@pytest.mark.parametrize(
    ("number", "age_days", "expected"),
    [
        (900, 0.0, (204.0, 942.0, 252.0)),
        (900, 1.0, (204.0, 3942.0, 252.0)),
        (900, -1.0, (204.0, 3942.0, 252.0)),
        (25933, 0.5, (146.0, 1762.0, 108.0)),
        (26113, 2.0, (3920.0, 13794.0, 3616.0)),
        (14129, 0.0, (1648.0, 2734.0, 2118.0)),
        (25924, 3.0, (718.0, 9864.0, 172.0)),
    ],
)
def test_sigmas_of_each_regime(
    number: int, age_days: float, expected: tuple[float, float, float]
) -> None:
    omm = BY_NUMBER[number]
    sigmas = MODEL.sigmas(omm, at(omm, age_days * 86400.0))
    assert isinstance(sigmas, RtnSigmas)
    assert sigmas == pytest.approx(expected, rel=1e-12)


def test_the_covariance_is_diagonal_in_rtn_with_no_velocity_block() -> None:
    omm = BY_NUMBER[900]
    time = at(omm, 86400.0)
    covariance = MODEL.covariance(omm, time)
    assert covariance.frame == "RTN"
    assert covariance.epoch is time
    assert covariance.repairs == ()
    expected = np.zeros((6, 6))
    expected[:3, :3] = np.diag(np.array([204.0, 3942.0, 252.0]) ** 2)
    np.testing.assert_allclose(covariance.matrix, expected, rtol=1e-12, atol=0.0)


def test_any_time_scale_gives_the_same_age() -> None:
    omm = BY_NUMBER[900]
    time = at(omm, 86400.0)
    assert MODEL.sigmas(omm, time.tdb) == pytest.approx(
        MODEL.sigmas(omm, time), rel=1e-12
    )


@given(elements, ages)
def test_the_covariance_is_a_valid_rtn_covariance(omm: Omm, age: float) -> None:
    covariance = MODEL.covariance(omm, at(omm, age))
    assert problems(covariance.matrix) == ()
    assert np.all(np.linalg.eigvalsh(covariance.position) > 0.0)
    assert np.all(covariance.matrix[3:, :] == 0.0)
    assert np.all(covariance.matrix[:, 3:] == 0.0)


@given(elements, ages)
def test_the_sigmas_do_not_depend_on_the_sign_of_the_age(omm: Omm, age: float) -> None:
    later = MODEL.sigmas(omm, at(omm, age))
    earlier = MODEL.sigmas(omm, at(omm, -age))
    assert earlier == pytest.approx(later, rel=1e-12)


@given(elements, ages, ages)
def test_only_the_transverse_sigma_grows_with_the_age(
    omm: Omm, first: float, second: float
) -> None:
    younger, older = sorted((first, second), key=abs)
    near = MODEL.sigmas(omm, at(omm, younger))
    far = MODEL.sigmas(omm, at(omm, older))
    assert far.transverse >= near.transverse
    assert (far.radial, far.normal) == (near.radial, near.normal)


@given(elements, ages, st.floats(min_value=0.1, max_value=10.0))
def test_the_safety_factor_scales_every_sigma(
    omm: Omm, age: float, factor: float
) -> None:
    time = at(omm, age)
    scaled = SyntheticCovarianceModel(safety_factor=factor).sigmas(omm, time)
    default = MODEL.sigmas(omm, time)
    assert scaled == pytest.approx(
        [factor / SAFETY_FACTOR * sigma for sigma in default], rel=1e-12
    )


CUSTOM = RegimeParameters(10.0, 20.0, 30.0, 0.0)

OVERRIDDEN = {
    regime: dataclasses.replace(MODEL, regimes={**MODEL.regimes, regime: CUSTOM})
    for regime in OrbitRegime
}
"""The default model with :data:`CUSTOM` parameters for one regime."""


@pytest.mark.parametrize("regime", OrbitRegime)
def test_each_regime_entry_gives_the_parameters_of_its_regime(
    regime: OrbitRegime,
) -> None:
    model = OVERRIDDEN[regime]
    assert model.parameters(regime) == CUSTOM
    for other in OrbitRegime:
        if other is not regime:
            assert model.parameters(other) == PUBLISHED_PARAMETERS[other]
    assert pickle.loads(pickle.dumps(model)) == model


def test_the_regimes_must_be_given_all() -> None:
    partial = {OrbitRegime.LEO: CUSTOM}
    with pytest.raises(ValueError, match="regimes must give every one of"):
        SyntheticCovarianceModel(regimes=partial)


@given(elements, ages, st.sampled_from(OrbitRegime))
def test_overriding_a_regime_changes_only_that_regime(
    omm: Omm, age: float, overridden: OrbitRegime
) -> None:
    time = at(omm, age)
    sigmas = OVERRIDDEN[overridden].sigmas(omm, time)
    if overridden is orbit_regime(omm):
        assert sigmas == (20.0, 40.0, 60.0)
    else:
        assert sigmas == MODEL.sigmas(omm, time)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("name", ["radial", "transverse", "normal"])
def test_the_sigmas_are_checked(name: str, bad: float) -> None:
    with pytest.raises(ValueError, match=f"{name} must be finite and > 0"):
        dataclasses.replace(PUBLISHED_PARAMETERS[OrbitRegime.LEO], **{name: bad})


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_the_safety_factor_is_checked(bad: float) -> None:
    with pytest.raises(ValueError, match="safety_factor must be finite and > 0"):
        SyntheticCovarianceModel(safety_factor=bad)


@pytest.mark.parametrize("bad", [-1e-9, math.inf, math.nan])
def test_the_rate_is_checked(bad: float) -> None:
    with pytest.raises(ValueError, match="transverse_rate must be finite and >= 0"):
        RegimeParameters(100.0, 200.0, 300.0, bad)


def test_parameters_are_stored_as_floats() -> None:
    parameters = RegimeParameters(100, 200, 300, 0)
    assert all(type(value) is float for value in dataclasses.astuple(parameters))
    assert type(SyntheticCovarianceModel(safety_factor=3).safety_factor) is float


def test_one_time_only() -> None:
    omm = BY_NUMBER[900]
    times = omm.epoch + np.array([0.0, 1.0]) * u.day
    with pytest.raises(ValueError, match=r"a single time, got shape \(2,\)"):
        MODEL.sigmas(omm, times)
    with pytest.raises(ValueError, match="a single time"):
        MODEL.covariance(omm, times)


def test_comments_describe_the_model_parameters_and_age() -> None:
    omm = BY_NUMBER[900]
    comments = MODEL.comments(omm, at(omm, 1.25 * 86400.0))
    epoch = omm.epoch.isot
    assert comments == (
        "Synthetic covariance: keepout GP element-set age model, an assumption",
        "sigma_T = f (sigma_T0 + k |t - epoch|); sigma_R, sigma_N constant",
        "Regime LEO: sigma_R0 0.102 km, sigma_T0 0.471 km, sigma_N0 0.126 km",
        "k 1.500 km/day, safety factor f 2",
        f"Element set epoch {epoch} UTC, age +1.250 days",
        "Sigmas: R 0.204 km, T 4.692 km, N 0.252 km",
    )
    earlier = MODEL.comments(omm, at(omm, -0.5 * 86400.0))
    assert earlier[4] == f"Element set epoch {epoch} UTC, age -0.500 days"


@given(elements, ages)
def test_comments_are_short_ascii_lines(omm: Omm, age: float) -> None:
    for line in MODEL.comments(omm, at(omm, age)):
        assert line.isascii()
        assert len(line) < 80


def test_models_compare_hash_and_pickle() -> None:
    copy = pickle.loads(pickle.dumps(MODEL))
    assert copy == MODEL
    assert hash(copy) == hash(MODEL)
    assert SyntheticCovarianceModel(safety_factor=1.0) != MODEL
