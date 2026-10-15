"""Validation cases built from NASA CARA reference data."""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, SupportsFloat, TypedDict

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
from keepout.sources.cara import Cell
from keepout.state import StateVector
from keepout.validation.criteria import (
    HALL,
    HALL_PC_SET,
    MONTE_CARLO,
    TWO_D,
    Criterion,
)
from keepout.validation.legacy import legacy_objects

__all__ = [
    "ALFANO",
    "ALFANO_3D",
    "ALFANO_MONTE_CARLO",
    "DILUTION_REFERENCES",
    "FRISBEE_REFERENCES",
    "MAX_PC_UNIT_TESTS",
    "MC_MIN_PC",
    "MC_SAMPLES",
    "MC_SUITE_ALPHA",
    "MC_TARGET_HITS",
    "NOMINAL_EPOCH",
    "PC_METHOD_SET",
    "PC_METHOD_SET_3D",
    "PC_METHOD_SET_MONTE_CARLO",
    "TWO_D_METHODS",
    "UNIT_TESTS",
    "AlfanoInput",
    "CaseOptions",
    "MonteCarloSizing",
    "PcMethodInput",
    "ValidationCase",
    "alfano_3d_cases",
    "alfano_cases",
    "alfano_monte_carlo_cases",
    "alfano_objects",
    "elrod_similar_to_3d_case",
    "hall_similar_to_2d_case",
    "load_alfano_inputs",
    "load_alfano_windows",
    "load_all",
    "load_hall_2d3d_cases",
    "load_max_pc_cases",
    "load_monte_carlo_cases",
    "load_non_pd_case",
    "load_pc_method_inputs",
    "monte_carlo_options",
    "pc_method_3d_cases",
    "pc_method_cases",
    "pc_method_monte_carlo_cases",
]

NOMINAL_EPOCH: Final = Time("2000-01-01T00:00:00", scale="utc")
"""Nominal epoch of the cases whose states have no calendar date (the Alfano
sheets and CARA's inline unit-test cases): inertial vectors at TCA; no Pc
depends on it."""

TWO_D_METHODS: Final[tuple[Method, ...]] = ("foster", "elrod")

ALFANO: Final = "Alfano (2009)"
ALFANO_3D: Final = "Alfano (2009), 3-D"
ALFANO_MONTE_CARLO: Final = "Alfano (2009), Monte Carlo"
PC_METHOD_SET: Final = "CARA Pc method test set"
PC_METHOD_SET_3D: Final = "CARA Pc method test set, 3-D"
PC_METHOD_SET_MONTE_CARLO: Final = "CARA Pc method test set, Monte Carlo"
UNIT_TESTS: Final = "CARA unit tests"
MAX_PC_UNIT_TESTS: Final = "CARA unit tests, maximum Pc"
"""Suite names (:attr:`ValidationCase.suite`), as the report titles them."""

MC_TARGET_HITS: Final = 400
"""Expected collisions a Monte Carlo case is sized for (about +-13% at 99%)."""

MC_SAMPLES: Final = (100_000, 4_000_000)
"""Bounds on the trials of one Monte Carlo case."""

MC_MIN_PC: Final = 1e-4
"""Smallest CARA Monte Carlo Pc compared: below it the trials needed exceed
:data:`MC_SAMPLES`."""

MC_SUITE_ALPHA: Final = 0.01
"""Chance that a correct keepout fails any of the ``n`` Monte Carlo cases: each
interval is at confidence ``1 - MC_SUITE_ALPHA / n`` (Bonferroni)."""


class MonteCarloSizing(TypedDict):
    """Trials and seed of a Monte Carlo case (:func:`monte_carlo_options`)."""

    samples: int
    seed: int


class CaseOptions(TypedDict, total=False):
    """Options a case passes to :func:`~keepout.pc.compute_pc`.

    ``window`` is passed to ``"hall3d"`` and ``"montecarlo"``, the others
    to ``"montecarlo"`` only.
    """

    window: tuple[float, float]
    samples: int
    seed: int
    confidence: float


_ACCEPTED: Final[Mapping[Method, frozenset[str]]] = {
    "hall3d": frozenset({"window"}),
    "montecarlo": CaseOptions.__optional_keys__,
}
"""The :class:`CaseOptions` each method takes; the others take none."""


@dataclass(frozen=True, eq=False, slots=True)
class ValidationCase:
    """One conjunction with a CARA reference Pc for some methods.

    Parameters
    ----------
    suite
        Name of the case collection, e.g. :data:`ALFANO`.
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
        Passed to :func:`~keepout.pc.compute_pc`; each must be taken by
        every method of the case (see :class:`CaseOptions`). Stored as a
        copy.
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
    options: CaseOptions = field(default_factory=CaseOptions)
    interval: tuple[float, float] | None = None
    informative: tuple[str, float] | None = None

    def __post_init__(self) -> None:
        for method in self.methods:
            refused = self.options.keys() - _ACCEPTED.get(method, frozenset())
            if refused:
                raise ValueError(
                    f"case {self.suite}/{self.case_id}: {method} takes no "
                    f"options {sorted(refused)}"
                )
        object.__setattr__(self, "options", self.options.copy())


def monte_carlo_options(suite: str, case_id: str, reference: float) -> MonteCarloSizing:
    """Trials and seed of a Monte Carlo case: about :data:`MC_TARGET_HITS`
    expected collisions within :data:`MC_SAMPLES`, and a seed fixed by the
    case's name so the report is reproducible."""
    wanted = MC_TARGET_HITS / reference if reference > 0.0 else MC_SAMPLES[1]
    samples = int(min(max(math.ceil(wanted), MC_SAMPLES[0]), MC_SAMPLES[1]))
    return {"samples": samples, "seed": zlib.crc32(f"{suite}/{case_id}".encode())}


def _vector(rows: Sequence[Sequence[Cell]], row: int) -> FloatArray:
    return np.array(rows[row - 1][:3], dtype=np.float64)


def _matrix(rows: Sequence[Sequence[Cell]], first_row: int) -> FloatArray:
    return np.array(
        [rows[index - 1][:6] for index in range(first_row, first_row + 6)],
        dtype=np.float64,
    )


def _gcrf_object(
    name: str, r: npt.ArrayLike, v: npt.ArrayLike, covariance: Covariance
) -> ObjectState:
    """An object at :data:`NOMINAL_EPOCH`."""
    return ObjectState(name, StateVector(NOMINAL_EPOCH, r, v, "GCRF"), covariance)


def alfano_objects(rows: Sequence[Sequence[Cell]]) -> tuple[ObjectState, ObjectState]:
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


def _field(row: Mapping[str, Cell], name: str) -> Cell:
    try:
        return row[name]
    except KeyError as exc:
        raise CaraFormatError(f"CARA table has no column {name!r}") from exc


def _number(row: Mapping[str, Cell], name: str) -> float:
    value = _field(row, name)
    error = f"column {name!r}: expected a number, got {value!r}"
    if not isinstance(value, str | SupportsFloat):
        raise CaraFormatError(error)
    try:
        return float(value)
    except ValueError as exc:
        raise CaraFormatError(error) from exc


@dataclass(frozen=True, eq=False, slots=True)
class AlfanoInput:
    """One Alfano (2009) case as CARA gives it.

    Parameters
    ----------
    number
        Case number, 1-12.
    conjunction
        The conjunction of its ``.xls`` sheet, with the HBR of ``results``.
    results
        Its row of CARA's results table.
    """

    number: int
    conjunction: Conjunction
    results: Mapping[str, Cell]


@dataclass(frozen=True, eq=False, slots=True)
class PcMethodInput:
    """One conjunction of the CARA Pc method test set.

    Parameters
    ----------
    conjunction_id
        CARA's identifier.
    conjunction
        The conjunction of its CDM, with the HBR of ``row``.
    row
        Its row of CARA's results table.
    """

    conjunction_id: str
    conjunction: Conjunction
    row: Mapping[str, Cell]


def load_alfano_inputs(cache: Path | None = None) -> list[AlfanoInput]:
    """Every Alfano (2009) case: CARA's results table and the ``.xls`` sheets."""
    results = {
        int(_number(row, "Case")): row
        for row in cara.load_csv(cara.fetch(cara.ALFANO_2009_RESULTS, cache=cache))
    }
    inputs: list[AlfanoInput] = []
    for number, file in enumerate(cara.ALFANO_2009_TCA, start=1):
        if number not in results:
            raise CaraFormatError(f"Alfano results have no row for case {number}")
        result = results[number]
        primary, secondary = alfano_objects(
            cara.load_xls_rows(cara.fetch(file, cache=cache))
        )
        conjunction = Conjunction(primary, secondary, _number(result, "HBR"))
        inputs.append(AlfanoInput(number, conjunction, result))
    return inputs


def load_alfano_windows(cache: Path | None = None) -> dict[int, float]:
    """Half-width [s] of each Alfano case's window, by case number:
    ``A09_FinalTime_s`` of CARA's supplementary table."""
    return {
        int(_number(row, "CaseNumber")): _number(row, "A09_FinalTime_s")
        for row in cara.load_xlsx(cara.fetch(cara.ALFANO_2009_SUPPLEMENT, cache=cache))
    }


def load_pc_method_inputs(cache: Path | None = None) -> list[PcMethodInput]:
    """Every conjunction of the CARA Pc method test set: table row and CDM."""
    inputs: list[PcMethodInput] = []
    for row in cara.load_xlsx(cara.fetch(cara.PC_METHOD_TABLE, cache=cache)):
        conjunction_id = str(_field(row, "Conjunction_ID"))
        if conjunction_id not in cara.PC_METHOD_CDMS:
            raise CaraFormatError(f"no pinned CDM for conjunction {conjunction_id}")
        path = cara.fetch(cara.PC_METHOD_CDMS[conjunction_id], cache=cache)
        conjunction = cdm.load(path).conjunction(hard_body_radius=_number(row, "HBR_m"))
        inputs.append(PcMethodInput(conjunction_id, conjunction, row))
    return inputs


def _window(windows: Mapping[int, float], number: int) -> tuple[float, float]:
    if number not in windows:
        raise CaraFormatError(f"Alfano supplement has no row for case {number}")
    return -windows[number], windows[number]


def _monte_carlo_case(
    suite: str,
    case_id: str,
    conjunction: Conjunction,
    reference: float,
    interval: tuple[float, float],
    window: tuple[float, float] | None = None,
) -> ValidationCase:
    """A case judged by keepout's Monte Carlo interval, sized for its reference.

    ``window`` is the Monte Carlo window; ``None`` sizes it automatically.
    """
    sizing = monte_carlo_options(suite, case_id, reference)
    options = CaseOptions(samples=sizing["samples"], seed=sizing["seed"])
    if window is not None:
        options["window"] = window
    return ValidationCase(
        suite,
        case_id,
        conjunction,
        reference,
        methods=("montecarlo",),
        criterion=MONTE_CARLO,
        options=options,
        interval=interval,
    )


def alfano_cases(alfano: Sequence[AlfanoInput]) -> list[ValidationCase]:
    """Alfano (2009) cases with a CARA 2-D reference (cases 1-11).

    Case 12 has zero relative velocity; CARA gives no 2-D value for it.
    """
    cases: list[ValidationCase] = []
    for item in alfano:
        reference = _number(item.results, "CARA_Pc2D")
        if math.isnan(reference):
            continue
        cases.append(
            ValidationCase(
                ALFANO,
                str(item.number),
                item.conjunction,
                reference,
                paper=_number(item.results, "A09_PcLinear100"),
            )
        )
    return cases


def alfano_3d_cases(
    alfano: Sequence[AlfanoInput], windows: Mapping[int, float]
) -> list[ValidationCase]:
    """Alfano (2009) cases 1-12 with CARA's 3D-Nc reference.

    The window is that of :func:`load_alfano_windows`, as in CARA's
    ``AlfanoTestCases_Pc3D.m``: cases 9 and 10 have the same states and
    differ only in it.
    """
    return [
        ValidationCase(
            ALFANO_3D,
            str(item.number),
            item.conjunction,
            _number(item.results, "CARA_Pc3D"),
            methods=("hall3d",),
            criterion=HALL,
            options={"window": _window(windows, item.number)},
        )
        for item in alfano
    ]


def alfano_monte_carlo_cases(
    alfano: Sequence[AlfanoInput], windows: Mapping[int, float]
) -> list[ValidationCase]:
    """Alfano (2009) cases 1-12 with CARA's two-body Monte Carlo reference."""
    return [
        _monte_carlo_case(
            ALFANO_MONTE_CARLO,
            str(item.number),
            item.conjunction,
            _number(item.results, "CARA_PcMC"),
            (
                _number(item.results, "CARA_PcMCLo95"),
                _number(item.results, "CARA_PcMCHi95"),
            ),
            _window(windows, item.number),
        )
        for item in alfano
    ]


def pc_method_cases(pc_method: Sequence[PcMethodInput]) -> list[ValidationCase]:
    """The 53 real conjunctions of the CARA Pc method test set.

    The reference is CARA's ``Pc2D``: states moved to the true TCA before
    integrating. keepout projects the relative position onto the encounter
    plane, which is the same straight-line closest approach.
    """
    return [
        ValidationCase(
            PC_METHOD_SET,
            item.conjunction_id,
            item.conjunction,
            _number(item.row, "Pc2D"),
        )
        for item in pc_method
    ]


def pc_method_3d_cases(pc_method: Sequence[PcMethodInput]) -> list[ValidationCase]:
    """The 53 CARA test conjunctions with CARA's ``Nc3D`` (Hall 3D-Nc).

    CARA's Coppola 3-D Pc is shown for information only: it is another
    method, and CARA's own table flags where it underestimates.
    """
    return [
        ValidationCase(
            PC_METHOD_SET_3D,
            item.conjunction_id,
            item.conjunction,
            _number(item.row, "Nc3D"),
            methods=("hall3d",),
            criterion=HALL_PC_SET,
            informative=("Coppola 3-D Pc", _number(item.row, "Pc3DCoppola")),
        )
        for item in pc_method
    ]


def pc_method_monte_carlo_cases(
    pc_method: Sequence[PcMethodInput],
) -> list[ValidationCase]:
    """Test-set conjunctions whose CARA Monte Carlo Pc is at least ``MC_MIN_PC``."""
    return [
        _monte_carlo_case(
            PC_METHOD_SET_MONTE_CARLO,
            item.conjunction_id,
            item.conjunction,
            reference,
            (_number(item.row, "PcSDMCLo"), _number(item.row, "PcSDMCHi")),
        )
        for item in pc_method
        if (reference := _number(item.row, "PcSDMC")) >= MC_MIN_PC
    ]


def load_non_pd_case(cache: Path | None = None) -> ValidationCase:
    """CDM with a non-positive-semi-definite covariance: Pc 0 within 1e-10.

    Reference and tolerance as in CARA's ``Pc2D_Foster_UnitTest.m``
    (``testNonPositiveDefinite``, HBR 20 m).
    """
    path = cara.fetch(cara.NON_PD_CDM, cache=cache)
    conjunction = cdm.load(path).conjunction(hard_body_radius=20.0)
    return ValidationCase(
        UNIT_TESTS,
        "non-PD covariance CDM",
        conjunction,
        0.0,
        criterion=Criterion(absolute=1e-10),
    )


_SIMILAR_STATES: Final = (
    (
        [-9.841950433215101e05, 3.932342044549424e05, 6.991223682230414e06],
        [4.883696742000000e03, 5.689086045000000e03, 3.665361590000000e02],
    ),
    (
        [-9.839696058965517e05, 3.936845951174244e05, 6.991219291625473e06],
        [1.509562687000000e03, 7.372938617000000e03, -1.492509430000000e02],
    ),
)
"""Primary and secondary ``(r, v)`` of CARA's inline ``testSimilarTo3D`` and
``testSimilarTo2D`` cases (ECI, m, m/s)."""


def _covariance(
    position: list[list[float]],
    cross: list[list[float]],
    velocity: list[list[float]],
) -> FloatArray:
    """A symmetric 6x6 covariance from its position, position-velocity and
    velocity blocks."""
    cross_block = np.array(cross)
    return np.block(
        [[np.array(position), cross_block], [cross_block.T, np.array(velocity)]]
    )


_SIMILAR_COVARIANCES: Final = (
    _covariance(
        [
            [4.976545641899520e04, 5.787130862568278e04, 3.370410320935015e03],
            [5.787130862568278e04, 6.730377643610841e04, 3.926542932121541e03],
            [3.370410320935015e03, 3.926542932121541e03, 2.461403197221289e02],
        ],
        [
            [1.137272273949272e01, -4.325472616114674e00, -8.009705480233521e01],
            [1.321992688238858e01, -5.035560720747812e00, -9.314985106902773e01],
            [7.586865834476763e-01, -3.077848629905763e-01, -5.434034460756914e00],
        ],
        [
            [2.608186227148725e-03, -9.804181796720670e-04, -1.829751672999786e-02],
            [-9.804181796720670e-04, 3.895883508545853e-04, 6.968892326415779e-03],
            [-1.829751672999786e-02, 6.968892326415779e-03, 1.289253320300791e-01],
        ],
    ),
    _covariance(
        [
            [4.246862551076427e04, 2.066374367781032e05, -5.011108933888592e03],
            [2.066374367781032e05, 1.005854717283451e06, -2.434876491048039e04],
            [-5.011108933888592e03, -2.434876491048039e04, 6.131274993037449e02],
        ],
        [
            [3.104606531932427e01, -1.201093683199582e01, -2.207975848324051e02],
            [1.510022508670080e02, -5.850063541467530e01, -1.074752763805685e03],
            [-3.667147183233717e00, 1.391769957262238e00, 2.601457791444154e01],
        ],
        [
            [2.272826228568773e-02, -8.778253314778023e-03, -1.613538091053610e-01],
            [-8.778253314778023e-03, 3.428801115804722e-03, 6.251148178133809e-02],
            [-1.613538091053610e-01, 6.251148178133809e-02, 1.148404222181769e00],
        ],
    ),
)
"""Primary and secondary 6x6 covariances of ``testSimilarTo2D`` (ECI, m,
m/s); ``testSimilarTo3D`` uses their position blocks."""


def _similar_conjunction(
    covariance: Callable[[FloatArray], Covariance],
) -> Conjunction:
    """The ``testSimilarTo*`` states with ``covariance`` of each 6x6 matrix,
    HBR 20 m."""
    primary, secondary = (
        _gcrf_object(name, r, v, covariance(matrix))
        for name, (r, v), matrix in zip(
            ("PRIMARY", "SECONDARY"),
            _SIMILAR_STATES,
            _SIMILAR_COVARIANCES,
            strict=True,
        )
    )
    return Conjunction(primary, secondary, 20.0)


def elrod_similar_to_3d_case() -> ValidationCase:
    """Inline case ``testSimilarTo3D`` of CARA's ``PcElrod_UnitTest.m``.

    Numbers transcribed from ``DistributedMatlab/ProbabilityOfCollision/
    UnitTests/PcElrod_UnitTest.m`` lines 208-232 at the pinned commit
    (ECI, m, m/s, HBR 20 m); the reference has 8 significant digits.
    """
    conjunction = _similar_conjunction(
        lambda matrix: Covariance.from_position(matrix[:3, :3], "GCRF", NOMINAL_EPOCH)
    )
    return ValidationCase(
        UNIT_TESTS, "Elrod testSimilarTo3D", conjunction, 1.0281653e-02
    )


def hall_similar_to_2d_case() -> ValidationCase:
    """Inline case ``testSimilarTo2D`` of CARA's ``Pc3D_Hall_UnitTest.m``.

    The states of Elrod's ``testSimilarTo3D`` (:func:`elrod_similar_to_3d_case`)
    with full 6x6 covariances, transcribed from
    ``DistributedMatlab/ProbabilityOfCollision/UnitTests/Pc3D_Hall_UnitTest.m``
    lines 131-145 at the pinned commit (ECI, m, m/s, HBR 20 m); the
    reference has 8 significant digits and CARA's tolerance is 1e-3.
    """
    conjunction = _similar_conjunction(
        lambda matrix: make_covariance(matrix, "GCRF", NOMINAL_EPOCH)
    )
    return ValidationCase(
        UNIT_TESTS,
        "Hall testSimilarTo2D",
        conjunction,
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
    conjunction = cdm.load(path).conjunction(hard_body_radius=20.0)
    return [
        ValidationCase(
            UNIT_TESTS,
            "Hall test2D3DDisagreement (2-D)",
            conjunction,
            2.2660816e-20,
        ),
        ValidationCase(
            UNIT_TESTS,
            "Hall test2D3DDisagreement (3-D)",
            conjunction,
            1.9271201e-05,
            methods=("hall3d",),
            criterion=HALL,
        ),
    ]


FRISBEE_REFERENCES: Final = ((5.0, 4.3e-5), (10.0, 1.71e-4))
"""HBR [m] and max Pc of ``test12`` and ``test13`` in CARA's
``FrisbeeMaxPc_UnitTest.m`` (Frisbee's published example, rounded to 1e-6,
hence CARA's absolute tolerance 5e-7), transcribed from
``DistributedMatlab/Maximum2DPc/UnitTests/FrisbeeMaxPc_UnitTest.m`` lines
641-698 (``test12``: values on lines 642-645) and 700-757 (``test13``: lines
701-704) at the pinned commit."""

DILUTION_REFERENCES: Final[Mapping[str, tuple[float, float]]] = {
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
    primary, secondary = legacy_objects(
        cara.fetch(cara.FRISBEE_CDM, cache=cache), known=(True, False)
    )
    cases = [
        ValidationCase(
            MAX_PC_UNIT_TESTS,
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
                MAX_PC_UNIT_TESTS,
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
    alfano = load_alfano_inputs(cache)
    windows = load_alfano_windows(cache)
    pc_method = load_pc_method_inputs(cache)
    return [
        *alfano_cases(alfano),
        *alfano_3d_cases(alfano, windows),
        *pc_method_cases(pc_method),
        *pc_method_3d_cases(pc_method),
        load_non_pd_case(cache),
        elrod_similar_to_3d_case(),
        hall_similar_to_2d_case(),
        *load_hall_2d3d_cases(cache),
        *load_max_pc_cases(cache),
    ]


def load_monte_carlo_cases(cache: Path | None = None) -> list[ValidationCase]:
    """The Monte Carlo validation cases (minutes to hours of computing), with
    the interval confidence of :data:`MC_SUITE_ALPHA`."""
    cara.prefetch(cara.MANIFEST, cache=cache)
    cases = [
        *alfano_monte_carlo_cases(
            load_alfano_inputs(cache), load_alfano_windows(cache)
        ),
        *pc_method_monte_carlo_cases(load_pc_method_inputs(cache)),
    ]
    confidence = 1.0 - MC_SUITE_ALPHA / max(len(cases), 1)
    return [
        replace(case, options={**case.options, "confidence": confidence})
        for case in cases
    ]
