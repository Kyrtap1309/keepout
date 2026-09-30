"""Validation cases built from NASA CARA reference data."""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from keepout import cdm
from keepout._types import FloatArray
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance, make_covariance
from keepout.errors import CaraFormatError
from keepout.objects import ObjectState
from keepout.pc.result import Method
from keepout.sources import cara
from keepout.state import StateVector
from keepout.validation.criteria import (
    HALL,
    HALL_PC_SET,
    MONTE_CARLO,
    TWO_D,
    Criterion,
)
from keepout.validation.legacy import legacy_objects

NOMINAL_EPOCH = Time("2000-01-01T00:00:00", scale="utc")
"""Nominal epoch of the cases whose states have no calendar date (the Alfano
sheets and CARA's inline unit-test cases): inertial vectors at TCA; no Pc
depends on it."""

TWO_D_METHODS: tuple[Method, ...] = ("foster", "elrod")

MC_TARGET_HITS = 400
"""Expected collisions a Monte Carlo case is sized for (about +-13% at 99%)."""

MC_SAMPLES = (100_000, 4_000_000)
"""Bounds on the trials of one Monte Carlo case."""

MC_MIN_PC = 1e-4
"""Smallest CARA Monte Carlo Pc compared: below it the trials needed exceed
:data:`MC_SAMPLES`."""


@dataclass(frozen=True, eq=False)
class ValidationCase:
    """One conjunction with a CARA reference Pc for some methods.

    Parameters
    ----------
    suite
        Name of the case collection, e.g. ``"Alfano (2009)"``.
    case_id
        Identifier within the suite.
    conjunction
        The conjunction to evaluate.
    reference
        CARA's Pc for these methods.
    methods
        keepout methods compared with the reference.
    criterion
        Pass/fail rule.
    paper
        Value published with the test case (Alfano 2009), if any.
    options
        Passed to :func:`~keepout.pc.compute_pc` (e.g. ``window``,
        ``samples``, ``seed``); stored read-only.
    interval
        CARA's own confidence interval (Monte Carlo references), if any.
    informative
        A labelled value shown next to the result without pass/fail.
    """

    suite: str
    case_id: str
    conjunction: Conjunction
    reference: float
    methods: tuple[Method, ...] = TWO_D_METHODS
    criterion: Criterion = TWO_D
    paper: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    interval: tuple[float, float] | None = None
    informative: tuple[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @property
    def monte_carlo(self) -> bool:
        """Whether the case is judged by a Monte Carlo interval."""
        return self.criterion.interval


def monte_carlo_options(suite: str, case_id: str, reference: float) -> dict[str, int]:
    """Trials and seed of a Monte Carlo case: about :data:`MC_TARGET_HITS`
    expected collisions within :data:`MC_SAMPLES`, and a seed fixed by the
    case's name so the report is reproducible."""
    wanted = MC_TARGET_HITS / reference if reference > 0.0 else MC_SAMPLES[1]
    samples = int(min(max(math.ceil(wanted), MC_SAMPLES[0]), MC_SAMPLES[1]))
    return {"samples": samples, "seed": zlib.crc32(f"{suite}/{case_id}".encode())}


def _vector(rows: Sequence[Sequence[Any]], row: int) -> FloatArray:
    return np.array(rows[row - 1][:3], dtype=np.float64)


def _matrix(rows: Sequence[Sequence[Any]], first_row: int) -> FloatArray:
    return np.array(
        [rows[index - 1][:6] for index in range(first_row, first_row + 6)],
        dtype=np.float64,
    )


def _gcrf_object(
    name: str, r: npt.ArrayLike, v: npt.ArrayLike, covariance: Covariance
) -> ObjectState:
    """An object at :data:`NOMINAL_EPOCH`."""
    return ObjectState(name, StateVector(NOMINAL_EPOCH, r, v, "GCRF"), covariance)


def alfano_objects(rows: Sequence[Sequence[Any]]) -> tuple[ObjectState, ObjectState]:
    """Primary and secondary from the rows of a ``CaseN data at TCA.xls`` sheet.

    Layout (1-based rows): primary position 5, velocity 9, covariance 13-18;
    secondary position 23, velocity 27, covariance 31-36. ECI, m, m/s.
    Covariances are repaired (symmetrised) if needed; see
    :attr:`~keepout.covariance.Covariance.repairs`.

    Raises
    ------
    CaraFormatError
        If the sheet does not have that layout.
    """
    try:
        pieces = [
            (_vector(rows, r), _vector(rows, v), _matrix(rows, c))
            for r, v, c in ((5, 9, 13), (23, 27, 31))
        ]
    except (IndexError, ValueError) as exc:
        raise CaraFormatError(f"unexpected Alfano sheet layout: {exc}") from exc
    primary, secondary = (
        _gcrf_object(name, r, v, make_covariance(matrix, "GCRF", NOMINAL_EPOCH))
        for name, (r, v, matrix) in zip(("PRIMARY", "SECONDARY"), pieces, strict=True)
    )
    return primary, secondary


def _field(row: Mapping[str, Any], name: str) -> Any:
    try:
        return row[name]
    except KeyError as exc:
        raise CaraFormatError(f"CARA table has no column {name!r}") from exc


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CaraFormatError(f"expected a number, got {value!r}") from exc


def load_alfano_cases(cache: Path | None = None) -> list[ValidationCase]:
    """Alfano (2009) cases with a CARA 2-D reference (cases 1-11).

    Case 12 has zero relative velocity; CARA gives no 2-D value for it.
    """
    results = _alfano_results(cache)
    cases: list[ValidationCase] = []
    for number, file in enumerate(cara.ALFANO_2009_TCA, start=1):
        if number not in results:
            raise CaraFormatError(f"Alfano results have no row for case {number}")
        result = results[number]
        reference = _float(_field(result, "CARA_Pc2D"))
        if math.isnan(reference):
            continue
        cases.append(
            ValidationCase(
                "Alfano (2009)",
                str(number),
                _alfano_conjunction(file, result, cache),
                reference,
                paper=_float(_field(result, "A09_PcLinear100")),
            )
        )
    return cases


def _alfano_results(cache: Path | None) -> dict[int, Mapping[str, Any]]:
    """CARA's Alfano (2009) results table, by case number."""
    return {
        int(_float(_field(row, "Case"))): row
        for row in cara.load_csv(cara.fetch(cara.ALFANO_2009_RESULTS, cache=cache))
    }


def _alfano_conjunction(
    file: cara.CaraFile, result: Mapping[str, Any], cache: Path | None
) -> Conjunction:
    """The conjunction of one Alfano case, with the HBR of its results row."""
    primary, secondary = alfano_objects(
        cara.load_xls_rows(cara.fetch(file, cache=cache))
    )
    return Conjunction(primary, secondary, _float(_field(result, "HBR")))


def load_pc_method_cases(cache: Path | None = None) -> list[ValidationCase]:
    """The 53 real conjunctions of the CARA Pc method test set.

    The reference is CARA's ``Pc2D``: states moved to the true TCA before
    integrating. keepout projects the relative position onto the encounter
    plane, which is the same straight-line closest approach.
    """
    return [
        ValidationCase(
            "CARA Pc method test set",
            conjunction_id,
            conjunction,
            _float(_field(row, "Pc2D")),
        )
        for conjunction_id, conjunction, row in _pc_method_rows(cache)
    ]


def load_non_pd_case(cache: Path | None = None) -> ValidationCase:
    """CDM with a non-positive-semi-definite covariance: Pc 0 within 1e-10.

    Reference and tolerance as in CARA's ``Pc2D_Foster_UnitTest.m``
    (``testNonPositiveDefinite``, HBR 20 m).
    """
    path = cara.fetch(cara.NON_PD_CDM, cache=cache)
    conjunction = Conjunction.from_cdm(cdm.load(path), hard_body_radius=20.0)
    return ValidationCase(
        "CARA unit tests",
        "non-PD covariance CDM",
        conjunction,
        0.0,
        criterion=Criterion(absolute=1e-10),
    )


def elrod_similar_to_3d_case() -> ValidationCase:
    """Inline case ``testSimilarTo3D`` of CARA's ``PcElrod_UnitTest.m``.

    Numbers transcribed from ``DistributedMatlab/ProbabilityOfCollision/
    UnitTests/PcElrod_UnitTest.m`` lines 208-232 at the pinned commit
    (ECI, m, m/s, HBR 20 m); the reference has 8 significant digits.
    """
    position_covariances = (
        [
            [4.976545641899520e04, 5.787130862568278e04, 3.370410320935015e03],
            [5.787130862568278e04, 6.730377643610841e04, 3.926542932121541e03],
            [3.370410320935015e03, 3.926542932121541e03, 2.461403197221289e02],
        ],
        [
            [4.246862551076427e04, 2.066374367781032e05, -5.011108933888592e03],
            [2.066374367781032e05, 1.005854717283451e06, -2.434876491048039e04],
            [-5.011108933888592e03, -2.434876491048039e04, 6.131274993037449e02],
        ],
    )
    states = (
        (
            [-9.841950433215101e05, 3.932342044549424e05, 6.991223682230414e06],
            [4.883696742000000e03, 5.689086045000000e03, 3.665361590000000e02],
        ),
        (
            [-9.839696058965517e05, 3.936845951174244e05, 6.991219291625473e06],
            [1.509562687000000e03, 7.372938617000000e03, -1.492509430000000e02],
        ),
    )
    primary, secondary = (
        _gcrf_object(
            name, r, v, Covariance.from_position(position, "GCRF", NOMINAL_EPOCH)
        )
        for name, (r, v), position in zip(
            ("PRIMARY", "SECONDARY"), states, position_covariances, strict=True
        )
    )
    return ValidationCase(
        "CARA unit tests",
        "Elrod testSimilarTo3D",
        Conjunction(primary, secondary, 20.0),
        1.0281653e-02,
    )


def _alfano_inputs(
    cache: Path | None,
) -> list[tuple[int, Conjunction, Mapping[str, Any], float]]:
    """Every Alfano case: number, conjunction, results row and window [s]."""
    results = _alfano_results(cache)
    windows = {
        int(_float(_field(row, "CaseNumber"))): _float(_field(row, "A09_FinalTime_s"))
        for row in cara.load_xlsx(cara.fetch(cara.ALFANO_2009_SUPPLEMENT, cache=cache))
    }
    inputs: list[tuple[int, Conjunction, Mapping[str, Any], float]] = []
    for number, file in enumerate(cara.ALFANO_2009_TCA, start=1):
        if number not in results or number not in windows:
            raise CaraFormatError(f"Alfano tables have no row for case {number}")
        result = results[number]
        conjunction = _alfano_conjunction(file, result, cache)
        inputs.append((number, conjunction, result, windows[number]))
    return inputs


def load_alfano_3d_cases(cache: Path | None = None) -> list[ValidationCase]:
    """Alfano (2009) cases 1-12 with CARA's 3D-Nc reference.

    The window is +-``A09_FinalTime_s`` of CARA's supplementary table, as in
    CARA's ``AlfanoTestCases_Pc3D.m``: cases 9 and 10 have the same states and
    differ only in it.
    """
    return [
        ValidationCase(
            "Alfano (2009), 3-D",
            str(number),
            conjunction,
            _float(_field(result, "CARA_Pc3D")),
            methods=("hall3d",),
            criterion=HALL,
            options={"window": (-window, window)},
        )
        for number, conjunction, result, window in _alfano_inputs(cache)
    ]


def load_alfano_monte_carlo_cases(cache: Path | None = None) -> list[ValidationCase]:
    """Alfano (2009) cases 1-12 with CARA's two-body Monte Carlo reference."""
    suite = "Alfano (2009), Monte Carlo"
    cases = []
    for number, conjunction, result, window in _alfano_inputs(cache):
        reference = _float(_field(result, "CARA_PcMC"))
        interval = (
            _float(_field(result, "CARA_PcMCLo95")),
            _float(_field(result, "CARA_PcMCHi95")),
        )
        options = {
            "window": (-window, window),
            **monte_carlo_options(suite, str(number), reference),
        }
        cases.append(
            ValidationCase(
                suite,
                str(number),
                conjunction,
                reference,
                methods=("montecarlo",),
                criterion=MONTE_CARLO,
                options=options,
                interval=interval,
            )
        )
    return cases


def _pc_method_rows(
    cache: Path | None,
) -> list[tuple[str, Conjunction, Mapping[str, Any]]]:
    """Every CARA Pc method test conjunction: ID, conjunction and table row."""
    table = cara.load_xlsx(cara.fetch(cara.PC_METHOD_TABLE, cache=cache))
    rows: list[tuple[str, Conjunction, Mapping[str, Any]]] = []
    for row in table:
        conjunction_id = str(_field(row, "Conjunction_ID"))
        if conjunction_id not in cara.PC_METHOD_CDMS:
            raise CaraFormatError(f"no pinned CDM for conjunction {conjunction_id}")
        path = cara.fetch(cara.PC_METHOD_CDMS[conjunction_id], cache=cache)
        conjunction = Conjunction.from_cdm(
            cdm.load(path), hard_body_radius=_float(_field(row, "HBR_m"))
        )
        rows.append((conjunction_id, conjunction, row))
    return rows


def load_pc_method_3d_cases(cache: Path | None = None) -> list[ValidationCase]:
    """The 53 CARA test conjunctions with CARA's ``Nc3D`` (Hall 3D-Nc).

    CARA's Coppola 3-D Pc is shown for information only: it is another
    method, and CARA's own table flags where it underestimates.
    """
    return [
        ValidationCase(
            "CARA Pc method test set, 3-D",
            conjunction_id,
            conjunction,
            _float(_field(row, "Nc3D")),
            methods=("hall3d",),
            criterion=HALL_PC_SET,
            informative=("Coppola 3-D Pc", _float(_field(row, "Pc3DCoppola"))),
        )
        for conjunction_id, conjunction, row in _pc_method_rows(cache)
    ]


def load_pc_method_monte_carlo_cases(
    cache: Path | None = None,
) -> list[ValidationCase]:
    """Test-set conjunctions whose CARA Monte Carlo Pc is at least ``MC_MIN_PC``."""
    suite = "CARA Pc method test set, Monte Carlo"
    cases = []
    for conjunction_id, conjunction, row in _pc_method_rows(cache):
        reference = _float(_field(row, "PcSDMC"))
        if reference < MC_MIN_PC:
            continue
        interval = (_float(_field(row, "PcSDMCLo")), _float(_field(row, "PcSDMCHi")))
        cases.append(
            ValidationCase(
                suite,
                conjunction_id,
                conjunction,
                reference,
                methods=("montecarlo",),
                criterion=MONTE_CARLO,
                options=monte_carlo_options(suite, conjunction_id, reference),
                interval=interval,
            )
        )
    return cases


def hall_similar_to_2d_case() -> ValidationCase:
    """Inline case ``testSimilarTo2D`` of CARA's ``Pc3D_Hall_UnitTest.m``.

    The states of Elrod's ``testSimilarTo3D`` (:func:`elrod_similar_to_3d_case`)
    with full 6x6 covariances, transcribed from
    ``DistributedMatlab/ProbabilityOfCollision/UnitTests/Pc3D_Hall_UnitTest.m``
    lines 131-145 at the pinned commit (ECI, m, m/s, HBR 20 m); the
    reference has 8 significant digits and CARA's tolerance is 1e-3.
    """
    covariances = (
        [
            [
                4.976545641899520e04,
                5.787130862568278e04,
                3.370410320935015e03,
                1.137272273949272e01,
                -4.325472616114674e00,
                -8.009705480233521e01,
            ],
            [
                5.787130862568278e04,
                6.730377643610841e04,
                3.926542932121541e03,
                1.321992688238858e01,
                -5.035560720747812e00,
                -9.314985106902773e01,
            ],
            [
                3.370410320935015e03,
                3.926542932121541e03,
                2.461403197221289e02,
                7.586865834476763e-01,
                -3.077848629905763e-01,
                -5.434034460756914e00,
            ],
            [
                1.137272273949272e01,
                1.321992688238858e01,
                7.586865834476763e-01,
                2.608186227148725e-03,
                -9.804181796720670e-04,
                -1.829751672999786e-02,
            ],
            [
                -4.325472616114674e00,
                -5.035560720747812e00,
                -3.077848629905763e-01,
                -9.804181796720670e-04,
                3.895883508545853e-04,
                6.968892326415779e-03,
            ],
            [
                -8.009705480233521e01,
                -9.314985106902773e01,
                -5.434034460756914e00,
                -1.829751672999786e-02,
                6.968892326415779e-03,
                1.289253320300791e-01,
            ],
        ],
        [
            [
                4.246862551076427e04,
                2.066374367781032e05,
                -5.011108933888592e03,
                3.104606531932427e01,
                -1.201093683199582e01,
                -2.207975848324051e02,
            ],
            [
                2.066374367781032e05,
                1.005854717283451e06,
                -2.434876491048039e04,
                1.510022508670080e02,
                -5.850063541467530e01,
                -1.074752763805685e03,
            ],
            [
                -5.011108933888592e03,
                -2.434876491048039e04,
                6.131274993037449e02,
                -3.667147183233717e00,
                1.391769957262238e00,
                2.601457791444154e01,
            ],
            [
                3.104606531932427e01,
                1.510022508670080e02,
                -3.667147183233717e00,
                2.272826228568773e-02,
                -8.778253314778023e-03,
                -1.613538091053610e-01,
            ],
            [
                -1.201093683199582e01,
                -5.850063541467530e01,
                1.391769957262238e00,
                -8.778253314778023e-03,
                3.428801115804722e-03,
                6.251148178133809e-02,
            ],
            [
                -2.207975848324051e02,
                -1.074752763805685e03,
                2.601457791444154e01,
                -1.613538091053610e-01,
                6.251148178133809e-02,
                1.148404222181769e00,
            ],
        ],
    )
    two_d = elrod_similar_to_3d_case().conjunction
    primary, secondary = (
        _gcrf_object(
            item.object_id,
            item.state.r,
            item.state.v,
            make_covariance(matrix, "GCRF", NOMINAL_EPOCH),
        )
        for item, matrix in zip(
            (two_d.primary, two_d.secondary), covariances, strict=True
        )
    )
    return ValidationCase(
        "CARA unit tests",
        "Hall testSimilarTo2D",
        Conjunction(primary, secondary, 20.0),
        1.0289973e-02,
        methods=("hall3d",),
        criterion=HALL,
    )


def load_hall_2d3d_cases(cache: Path | None = None) -> list[ValidationCase]:
    """``test2D3DDisagreementCase`` of CARA's ``Pc3D_Hall_UnitTest.m``.

    ``OmitronTestCase_Test08_3DNc.cdm`` with HBR 20 m: CARA's 2-D Pc is
    2.2660816e-20 and its 3-D Pc 1.9271201e-05 (tolerance 1e-3), transcribed
    from ``DistributedMatlab/ProbabilityOfCollision/UnitTests/
    Pc3D_Hall_UnitTest.m`` lines 107-122 (values on lines 108-111) at the
    pinned commit.
    """
    path = cara.fetch(cara.HALL_3D_CDM, cache=cache)
    conjunction = Conjunction.from_cdm(cdm.load(path), hard_body_radius=20.0)
    return [
        ValidationCase(
            "CARA unit tests",
            "Hall test2D3DDisagreement (2-D)",
            conjunction,
            2.2660816e-20,
        ),
        ValidationCase(
            "CARA unit tests",
            "Hall test2D3DDisagreement (3-D)",
            conjunction,
            1.9271201e-05,
            methods=("hall3d",),
            criterion=HALL,
        ),
    ]


FRISBEE_REFERENCES = ((5.0, 4.3e-5), (10.0, 1.71e-4))
"""HBR [m] and max Pc of ``test12`` and ``test13`` in CARA's
``FrisbeeMaxPc_UnitTest.m`` (Frisbee's published example, rounded to 1e-6,
hence CARA's absolute tolerance 5e-7), transcribed from
``DistributedMatlab/Maximum2DPc/UnitTests/FrisbeeMaxPc_UnitTest.m`` lines
641-698 (``test12``: values on lines 642-645) and 700-757 (``test13``: lines
701-704) at the pinned commit."""

DILUTION_REFERENCES = {
    "Test01_HighPc": (20.0, 0.999),
    "Test02_MaxRadialSigma": (100.0, 0.999),
    "Test03_MaxIntrackSigma": (100.0, 9.10e-2),
    "Test05_MinMiss": (20.0, 0.999),
}
"""HBR [m] and max Pc of ``test01``, ``test02``, ``test03`` and ``test05`` in
CARA's ``DilutionMaxPc_UnitTest.m`` (tolerance 5e-3; the CDM of ``test04`` is
not in the repository), transcribed from
``DistributedMatlab/Maximum2DPc/UnitTests/DilutionMaxPc_UnitTest.m`` lines
11-122, 124-219, 221-316 and 419-514 (values on lines 12-15, 125-128,
222-225 and 420-423) at the pinned commit."""


def load_max_pc_cases(cache: Path | None = None) -> list[ValidationCase]:
    """CARA's maximum-Pc unit tests (Frisbee and dilution).

    The CDMs are read with :func:`~keepout.validation.legacy.legacy_objects`:
    they do not conform to CCSDS. Frisbee's secondary covariance is unknown,
    as CARA's test passes zeros for it.
    """
    suite = "CARA unit tests, maximum Pc"
    primary, secondary = legacy_objects(
        cara.fetch(cara.FRISBEE_CDM, cache=cache), known=(True, False)
    )
    cases = [
        ValidationCase(
            suite,
            f"Frisbee example, HBR {radius:g} m",
            Conjunction(primary, secondary, radius),
            reference,
            methods=("frisbee",),
            criterion=Criterion(absolute=5e-7),
        )
        for radius, reference in FRISBEE_REFERENCES
    ]
    for name, (radius, reference) in DILUTION_REFERENCES.items():
        objects = legacy_objects(cara.fetch(cara.DILUTION_CDMS[name], cache=cache))
        cases.append(
            ValidationCase(
                suite,
                f"Omitron {name}",
                Conjunction(*objects, radius),
                reference,
                methods=("dilution",),
                criterion=Criterion(relative=5e-3),
            )
        )
    return cases


def load_all(cache: Path | None = None) -> list[ValidationCase]:
    """Every validation case except Monte Carlo, downloading CARA files as needed."""
    cara.prefetch(cara.MANIFEST, cache=cache)
    return [
        *load_alfano_cases(cache),
        *load_alfano_3d_cases(cache),
        *load_pc_method_cases(cache),
        *load_pc_method_3d_cases(cache),
        load_non_pd_case(cache),
        elrod_similar_to_3d_case(),
        hall_similar_to_2d_case(),
        *load_hall_2d3d_cases(cache),
        *load_max_pc_cases(cache),
    ]


def load_monte_carlo_cases(cache: Path | None = None) -> list[ValidationCase]:
    """The Monte Carlo validation cases (minutes to hours of computing)."""
    cara.prefetch(cara.MANIFEST, cache=cache)
    return [
        *load_alfano_monte_carlo_cases(cache),
        *load_pc_method_monte_carlo_cases(cache),
    ]
