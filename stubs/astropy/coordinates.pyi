# The part of astropy.coordinates keepout uses: Cartesian states with a
# velocity, in a frame at given times, transformed to another frame. The
# stub narrows astropy's unions: ``.cartesian`` is Cartesian, and so is the
# differential it carries (converted with it).
from typing import TypeVar

from astropy.time import Time
from astropy.units import Quantity

_Frame = TypeVar("_Frame", bound=BaseCoordinateFrame)

class CartesianDifferential:
    def __init__(self, d_x: Quantity, /) -> None: ...
    @property
    def d_xyz(self) -> Quantity: ...

class CartesianRepresentation:
    def __init__(
        self, x: Quantity, /, *, differentials: CartesianDifferential | None = None
    ) -> None: ...
    @property
    def xyz(self) -> Quantity: ...
    @property
    def differentials(self) -> dict[str, CartesianDifferential]: ...

class BaseCoordinateFrame:
    def __init__(
        self,
        representation: CartesianRepresentation | None = None,
        /,
        *,
        obstime: Time | None = None,
    ) -> None: ...
    def transform_to(self, frame: _Frame, /) -> _Frame: ...
    @property
    def cartesian(self) -> CartesianRepresentation: ...

class GCRS(BaseCoordinateFrame): ...
class ITRS(BaseCoordinateFrame): ...
class TEME(BaseCoordinateFrame): ...
