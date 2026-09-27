"""A small validation case for report tests (no CARA download)."""

import numpy as np
from astropy.time import Time

from keepout._types import StateVector
from keepout.conjunction import Conjunction
from keepout.covariance import Covariance
from keepout.objects import ObjectState
from keepout.validation.cases import ValidationCase

EPOCH = Time("2010-03-13T22:37:52.618", scale="utc")


def _object(object_id: str, r: list[float], v: list[float]) -> ObjectState:
    matrix = np.diag([100.0, 100.0, 100.0, 1e-4, 1e-4, 1e-4])
    return ObjectState(
        object_id, StateVector(EPOCH, r, v, "GCRF"), Covariance(matrix, "GCRF", EPOCH)
    )


def case_with_reference(
    reference: float,
    *,
    case_id: str = "case",
    absolute_tolerance: float | None = None,
) -> ValidationCase:
    """Head-on LEO case (miss 1 km, sigma 10 m, HBR 5 m: Pc underflows to 0)."""
    primary = _object("PRI", [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
    secondary = _object("SEC", [7001e3, 0.0, 0.0], [0.0, -7500.0, 0.0])
    return ValidationCase(
        "test",
        case_id,
        Conjunction(primary, secondary, 5.0),
        reference,
        absolute_tolerance=absolute_tolerance,
    )
