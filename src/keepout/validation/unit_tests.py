"""Numbers transcribed from NASA CARA's MATLAB unit tests: states,
covariances, hard-body radii and reference Pc of the inline cases.

The files are those of ``DistributedMatlab`` at the pinned commit
(:data:`keepout.sources.cara.MANIFEST`); each value cites its lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from keepout.conjunction import Conjunction
from keepout.objects import ObjectState
from keepout.state import StateVector
from keepout.validation.cara_inputs import NOMINAL_EPOCH

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from keepout._types import FloatArray
    from keepout.covariance import Covariance

__all__ = [
    "DILUTION_REFERENCES",
    "ELROD_SIMILAR_TO_3D_PC",
    "FRISBEE_REFERENCES",
    "HALL_2D3D_PC_2D",
    "HALL_2D3D_PC_3D",
    "HALL_SIMILAR_TO_2D_PC",
    "SIMILAR_COVARIANCES",
    "SIMILAR_STATES",
    "UNIT_TEST_HBR",
    "similar_conjunction",
]

UNIT_TEST_HBR: Final = 20.0
"""Hard-body radius [m] of CARA's ``Pc2D_Foster``, ``PcElrod`` and
``Pc3D_Hall`` unit tests used here."""

SIMILAR_STATES: Final = (
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


SIMILAR_COVARIANCES: Final = (
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

ELROD_SIMILAR_TO_3D_PC: Final = 1.0281653e-02
"""CARA's Pc of ``testSimilarTo3D`` in ``UnitTests/PcElrod_UnitTest.m``
(lines 208-232; 8 significant digits)."""

HALL_SIMILAR_TO_2D_PC: Final = 1.0289973e-02
"""CARA's Pc of ``testSimilarTo2D`` in ``UnitTests/Pc3D_Hall_UnitTest.m``
(lines 131-145; 8 significant digits, CARA's tolerance 1e-3)."""

HALL_2D3D_PC_2D: Final = 2.2660816e-20
"""CARA's 2-D Pc of ``test2D3DDisagreementCase`` in
``UnitTests/Pc3D_Hall_UnitTest.m`` (lines 107-122, values on lines
108-111)."""

HALL_2D3D_PC_3D: Final = 1.9271201e-05
"""CARA's 3-D Pc of the same case (tolerance 1e-3)."""

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


def similar_conjunction(
    covariance: Callable[[FloatArray], Covariance],
) -> Conjunction:
    """The ``testSimilarTo*`` states at :data:`NOMINAL_EPOCH` with
    ``covariance`` of each 6x6 matrix, HBR :data:`UNIT_TEST_HBR`.

    Parameters
    ----------
    covariance
        What makes a :class:`~keepout.covariance.Covariance` of a 6x6 GCRF matrix [m,
        m/s], e.g. with a repair policy.

    Returns
    -------
    Conjunction
        The conjunction.
    """
    primary, secondary = (
        ObjectState(name, StateVector(NOMINAL_EPOCH, r, v, "GCRF"), covariance(matrix))
        for name, (r, v), matrix in zip(
            ("PRIMARY", "SECONDARY"),
            SIMILAR_STATES,
            SIMILAR_COVARIANCES,
            strict=True,
        )
    )
    return Conjunction(primary, secondary, UNIT_TEST_HBR)
