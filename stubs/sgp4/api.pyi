from collections.abc import Sequence
from typing import Final, Self, TypeAlias

import numpy as np
import numpy.typing as npt

SGP4_ERRORS: Final[dict[int, str]]
WGS72OLD: Final[int]
WGS72: Final[int]
WGS84: Final[int]

_Vector: TypeAlias = tuple[float, float, float]

class Satrec:
    # The compiled SGP4 satellite record (sgp4.wrapper.Satrec).

    # Elements and identification, as initialised.
    satnum: int
    satnum_str: str
    classification: str
    intldesg: str
    ephtype: int
    elnum: int
    revnum: int
    epochyr: int
    epochdays: float
    jdsatepoch: float
    jdsatepochF: float
    no_kozai: float
    ecco: float
    inclo: float
    nodeo: float
    argpo: float
    mo: float
    bstar: float
    ndot: float
    nddot: float
    # Values sgp4init computes, and the gravity constants.
    method: str
    error: int
    a: float
    alta: float
    altp: float
    j2: float
    j3oj2: float
    radiusearthkm: float
    xke: float
    # Mean elements of the latest propagation.
    am: float
    em: float
    def __init__(self) -> None: ...
    def sgp4init(
        self,
        whichconst: int,
        opsmode: str,
        satnum: int,
        epoch: float,
        bstar: float,
        ndot: float,
        nddot: float,
        ecco: float,
        argpo: float,
        inclo: float,
        mo: float,
        no_kozai: float,
        nodeo: float,
    ) -> None: ...
    @classmethod
    def twoline2rv(cls, line1: str, line2: str, whichconst: int = ...) -> Self: ...
    def sgp4(self, jd: float, fr: float) -> tuple[int, _Vector, _Vector]: ...
    def sgp4_tsince(self, tsince: float) -> tuple[int, _Vector, _Vector]: ...
    def sgp4_array(
        self, jd: npt.NDArray[np.float64], fr: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.uint8], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]: ...

class SatrecArray:
    # The compiled SGP4 over several records (sgp4.wrapper.SatrecArray).

    def __init__(self, satrecs: Sequence[Satrec]) -> None: ...
    def __len__(self) -> int: ...
    def sgp4(
        self, jd: npt.NDArray[np.float64], fr: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.uint8], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]: ...
