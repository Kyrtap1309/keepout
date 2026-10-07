from typing import Final

WGS72OLD: Final = 0
WGS72: Final = 1
WGS84: Final = 2

_Vector = tuple[float, float, float]

class Satrec:
    """sgp4's pure-Python SGP4 satellite record."""

    # Every attribute is set by sgp4init (``__slots__`` at run time); these
    # are the ones read by name, the rest through ``getattr``.
    isimp: int
    no_unkozai: float
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
    def sgp4_tsince(self, tsince: float) -> tuple[int, _Vector, _Vector]: ...
