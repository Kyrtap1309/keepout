# The IERS settings and tables the tests keep offline.
from contextlib import AbstractContextManager
from typing import Self

class _Conf:
    def set_temp(self, attr: str, value: object) -> AbstractContextManager[None]: ...

conf: _Conf

class IERS_Auto:
    @classmethod
    def open(cls, *args: object, **kwargs: object) -> Self: ...

class IERS_B:
    @classmethod
    def open(cls, *args: object, **kwargs: object) -> Self: ...
