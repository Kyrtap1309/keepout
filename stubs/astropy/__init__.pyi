# Only astropy.time is typed here (keepout's Time-heavy code); every other
# astropy module stays untyped, as without these stubs.
from typing import Any

def __getattr__(name: str) -> Any: ...
