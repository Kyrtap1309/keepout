from sgp4.model import Satrec

def _dpper(
    satrec: Satrec,
    inclo: float,
    init: str,
    ep: float,
    inclp: float,
    nodep: float,
    argpp: float,
    mp: float,
    opsmode: str,
) -> tuple[float, float, float, float, float]: ...
