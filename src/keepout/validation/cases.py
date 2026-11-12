"""Validation cases built from NASA CARA reference data.

The inputs come from CARA's files (:mod:`keepout.validation.cara_inputs`)
and from the numbers of its unit tests (:mod:`keepout.validation.unit_tests`).
"""

from __future__ import annotations

import enum
import math
import zlib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, TypedDict

from keepout import cdm
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance, make_covariance
from keepout.errors import CaraFormatError
from keepout.pc.api import METHODS, require_options
from keepout.sources import cara
from keepout.sources.cara_legacy import load_legacy_objects
from keepout.validation.cara_inputs import (
    NOMINAL_EPOCH,
    AlfanoInput,
    PcMethodInput,
    column_number,
    load_alfano_inputs,
    load_alfano_windows,
    load_pc_method_inputs,
)
from keepout.validation.criteria import (
    HALL,
    HALL_PC_SET,
    MONTE_CARLO,
    TWO_D,
    BoundCriterion,
    Criterion,
)
from keepout.validation.unit_tests import (
    DILUTION_REFERENCES,
    ELROD_SIMILAR_TO_3D_PC,
    FRISBEE_REFERENCES,
    HALL_2D3D_PC_2D,
    HALL_2D3D_PC_3D,
    HALL_SIMILAR_TO_2D_PC,
    UNIT_TEST_HBR,
    similar_conjunction,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from keepout.pc.result import Method

__all__ = [
    "MC_MIN_PC",
    "MC_SAMPLES",
    "MC_SUITE_ALPHA",
    "MC_TARGET_HITS",
    "TWO_D_METHODS",
    "CaseOptions",
    "MonteCarloSizing",
    "Suite",
    "ValidationCase",
    "alfano_3d_cases",
    "alfano_cases",
    "alfano_monte_carlo_cases",
    "elrod_similar_to_3d_case",
    "hall_similar_to_2d_case",
    "load_all",
    "load_hall_2d3d_cases",
    "load_max_pc_cases",
    "load_monte_carlo_cases",
    "load_non_pd_case",
    "monte_carlo_options",
    "pc_method_3d_cases",
    "pc_method_cases",
    "pc_method_monte_carlo_cases",
]

TWO_D_METHODS: Final[tuple[Method, ...]] = tuple(
    name for name, spec in METHODS.items() if spec.kind == "2d"
)
"""The 2-D methods (:data:`~keepout.pc.METHODS`), which the 2-D cases compare."""


class Suite(enum.StrEnum):
    """The collections of validation cases (:attr:`ValidationCase.suite`);
    each value is the suite's name, as the report titles it."""

    ALFANO = "Alfano (2009)"
    ALFANO_3D = "Alfano (2009), 3-D"
    ALFANO_MONTE_CARLO = "Alfano (2009), Monte Carlo"
    PC_METHOD_SET = "CARA Pc method test set"
    PC_METHOD_SET_3D = "CARA Pc method test set, 3-D"
    PC_METHOD_SET_MONTE_CARLO = "CARA Pc method test set, Monte Carlo"
    UNIT_TESTS = "CARA unit tests"
    MAX_PC_UNIT_TESTS = "CARA unit tests, maximum Pc"


MC_TARGET_HITS: Final = 400
"""Expected collisions a Monte Carlo case is sized for (about +-13% at 99%)."""

MC_SAMPLES: Final = (100_000, 4_000_000)
"""Bounds on the samples of one Monte Carlo case."""

MC_MIN_PC: Final = 1e-4
"""Smallest CARA Monte Carlo Pc compared: below it the trials needed exceed
:data:`MC_SAMPLES`."""

MC_SUITE_ALPHA: Final = 0.01
"""Chance that a correct keepout fails any of the ``n`` Monte Carlo cases: each
interval is at confidence ``1 - MC_SUITE_ALPHA / n`` (Bonferroni)."""


class MonteCarloSizing(TypedDict):
    """Samples and seed of a Monte Carlo case (:func:`monte_carlo_options`)."""

    samples: int
    seed: int


class CaseOptions(TypedDict, total=False):
    """Options a case passes to :func:`~keepout.pc.compute_pc`.

    ``"hall3d"`` takes ``window``, ``"montecarlo"`` every one (see
    :attr:`~keepout.pc.MethodSpec.options`).
    """

    window: tuple[float, float]
    samples: int
    seed: int
    confidence: float


@dataclass(frozen=True, eq=False, slots=True)
class ValidationCase:
    """One conjunction with a CARA reference Pc for some methods.

    Parameters
    ----------
    suite
        The case collection, e.g. :attr:`Suite.ALFANO`.
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
        every method of the case (its :attr:`~keepout.pc.MethodSpec.options`).
        Stored as a copy.
    reference_interval
        CARA's own confidence interval (Monte Carlo references), if any.
    informative
        A labelled value shown next to the result without pass/fail.
    """

    suite: Suite
    case_id: str
    conjunction: Conjunction
    reference: float
    methods: tuple[Method, ...] = TWO_D_METHODS
    criterion: Criterion = TWO_D
    paper: float | None = None
    options: CaseOptions = field(default_factory=CaseOptions)
    reference_interval: tuple[float, float] | None = None
    informative: tuple[str, float] | None = None

    def __post_init__(self) -> None:
        for method in self.methods:
            try:
                require_options(method, self.options)
            except TypeError as exc:
                raise ValueError(f"case {self.suite}/{self.case_id}: {exc}") from exc
        object.__setattr__(self, "options", self.options.copy())


def monte_carlo_options(
    suite: Suite, case_id: str, reference: float
) -> MonteCarloSizing:
    """Samples and seed of a Monte Carlo case: about :data:`MC_TARGET_HITS`
    expected collisions within :data:`MC_SAMPLES`, and a seed fixed by the
    case's name so the report is reproducible."""
    wanted = MC_TARGET_HITS / reference if reference > 0.0 else MC_SAMPLES[1]
    samples = int(min(max(math.ceil(wanted), MC_SAMPLES[0]), MC_SAMPLES[1]))
    return {"samples": samples, "seed": zlib.crc32(f"{suite}/{case_id}".encode())}


def _window(windows: Mapping[int, float], number: int) -> tuple[float, float]:
    if number not in windows:
        raise CaraFormatError(f"Alfano supplement has no row for case {number}")
    return -windows[number], windows[number]


def _monte_carlo_case(
    suite: Suite,
    case_id: str,
    conjunction: Conjunction,
    reference: float,
    reference_interval: tuple[float, float],
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
        reference_interval=reference_interval,
    )


def alfano_cases(alfano: Sequence[AlfanoInput]) -> list[ValidationCase]:
    """Alfano (2009) cases with a CARA 2-D reference (cases 1-11).

    Case 12 has zero relative velocity; CARA gives no 2-D value for it.
    """
    cases: list[ValidationCase] = []
    for item in alfano:
        reference = column_number(item.results, "CARA_Pc2D")
        if math.isnan(reference):
            continue
        cases.append(
            ValidationCase(
                Suite.ALFANO,
                str(item.number),
                item.conjunction,
                reference,
                paper=column_number(item.results, "A09_PcLinear100"),
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
            Suite.ALFANO_3D,
            str(item.number),
            item.conjunction,
            column_number(item.results, "CARA_Pc3D"),
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
            Suite.ALFANO_MONTE_CARLO,
            str(item.number),
            item.conjunction,
            column_number(item.results, "CARA_PcMC"),
            (
                column_number(item.results, "CARA_PcMCLo95"),
                column_number(item.results, "CARA_PcMCHi95"),
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
            Suite.PC_METHOD_SET,
            item.conjunction_id,
            item.conjunction,
            column_number(item.row, "Pc2D"),
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
            Suite.PC_METHOD_SET_3D,
            item.conjunction_id,
            item.conjunction,
            column_number(item.row, "Nc3D"),
            methods=("hall3d",),
            criterion=HALL_PC_SET,
            informative=("Coppola 3-D Pc", column_number(item.row, "Pc3DCoppola")),
        )
        for item in pc_method
    ]


def pc_method_monte_carlo_cases(
    pc_method: Sequence[PcMethodInput],
) -> list[ValidationCase]:
    """Test-set conjunctions whose CARA Monte Carlo Pc is at least ``MC_MIN_PC``."""
    return [
        _monte_carlo_case(
            Suite.PC_METHOD_SET_MONTE_CARLO,
            item.conjunction_id,
            item.conjunction,
            reference,
            (column_number(item.row, "PcSDMCLo"), column_number(item.row, "PcSDMCHi")),
        )
        for item in pc_method
        if (reference := column_number(item.row, "PcSDMC")) >= MC_MIN_PC
    ]


def load_non_pd_case(cache: Path | None = None) -> ValidationCase:
    """CDM with a non-positive-semi-definite covariance: Pc 0 within 1e-10.

    Reference and tolerance as in CARA's ``Pc2D_Foster_UnitTest.m``
    (``testNonPositiveDefinite``, HBR 20 m).
    """
    path = cara.fetch(cara.NON_PD_CDM, cache=cache)
    conjunction = cdm.load(path).conjunction(combined_hard_body_radius=UNIT_TEST_HBR)
    return ValidationCase(
        Suite.UNIT_TESTS,
        "non-PD covariance CDM",
        conjunction,
        0.0,
        criterion=BoundCriterion(absolute=1e-10),
    )


def elrod_similar_to_3d_case() -> ValidationCase:
    """Inline case ``testSimilarTo3D`` of CARA's ``PcElrod_UnitTest.m``.

    Numbers transcribed from ``DistributedMatlab/ProbabilityOfCollision/
    UnitTests/PcElrod_UnitTest.m`` lines 208-232 at the pinned commit
    (ECI, m, m/s, HBR 20 m); the reference has 8 significant digits.
    """
    conjunction = similar_conjunction(
        lambda matrix: Covariance.from_position(matrix[:3, :3], "GCRF", NOMINAL_EPOCH)
    )
    return ValidationCase(
        Suite.UNIT_TESTS, "Elrod testSimilarTo3D", conjunction, ELROD_SIMILAR_TO_3D_PC
    )


def hall_similar_to_2d_case() -> ValidationCase:
    """Inline case ``testSimilarTo2D`` of CARA's ``Pc3D_Hall_UnitTest.m``.

    The states of Elrod's ``testSimilarTo3D`` (:func:`elrod_similar_to_3d_case`)
    with full 6x6 covariances, transcribed from
    ``DistributedMatlab/ProbabilityOfCollision/UnitTests/Pc3D_Hall_UnitTest.m``
    lines 131-145 at the pinned commit (ECI, m, m/s, HBR 20 m); the
    reference has 8 significant digits and CARA's tolerance is 1e-3.
    """
    conjunction = similar_conjunction(
        lambda matrix: make_covariance(matrix, "GCRF", NOMINAL_EPOCH)
    )
    return ValidationCase(
        Suite.UNIT_TESTS,
        "Hall testSimilarTo2D",
        conjunction,
        HALL_SIMILAR_TO_2D_PC,
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
    conjunction = cdm.load(path).conjunction(combined_hard_body_radius=UNIT_TEST_HBR)
    return [
        ValidationCase(
            Suite.UNIT_TESTS,
            "Hall test2D3DDisagreement (2-D)",
            conjunction,
            HALL_2D3D_PC_2D,
        ),
        ValidationCase(
            Suite.UNIT_TESTS,
            "Hall test2D3DDisagreement (3-D)",
            conjunction,
            HALL_2D3D_PC_3D,
            methods=("hall3d",),
            criterion=HALL,
        ),
    ]


def load_max_pc_cases(cache: Path | None = None) -> list[ValidationCase]:
    """CARA's maximum-Pc unit tests (Frisbee and dilution).

    The CDMs are read with :func:`~keepout.sources.cara_legacy.load_legacy_objects`:
    they do not conform to CCSDS. Frisbee's secondary covariance is unknown,
    as CARA's test passes zeros for it.
    """
    primary, secondary = load_legacy_objects(
        cara.fetch(cara.FRISBEE_CDM, cache=cache), known=(True, False)
    )
    cases = [
        ValidationCase(
            Suite.MAX_PC_UNIT_TESTS,
            f"Frisbee example, HBR {radius:g} m",
            Conjunction(primary, secondary, radius),
            reference,
            methods=("frisbee",),
            criterion=BoundCriterion(absolute=5e-7),
        )
        for radius, reference in FRISBEE_REFERENCES
    ]
    for name, (radius, reference) in DILUTION_REFERENCES.items():
        objects = load_legacy_objects(cara.fetch(cara.DILUTION_CDMS[name], cache=cache))
        cases.append(
            ValidationCase(
                Suite.MAX_PC_UNIT_TESTS,
                f"Omitron {name}",
                Conjunction(*objects, radius),
                reference,
                methods=("dilution",),
                criterion=BoundCriterion(relative=5e-3),
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
