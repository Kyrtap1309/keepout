"""The errors of the Pc methods that callers tell apart: an unknown
covariance (:class:`MissingCovariance`) and an orbit the 3-D methods cannot
use (:class:`UnusableOrbit`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from keepout.errors import CovarianceError, GeometryError

if TYPE_CHECKING:
    from keepout.encounter import Role
    from keepout.pc.result import Method

__all__ = ["MissingCovariance", "UnusableOrbit", "require_known_covariance"]


class MissingCovariance(CovarianceError):
    """Covariances a Pc method needs are unknown.

    Parameters
    ----------
    roles
        The objects whose covariance is unknown.
    method
        The method that needs them, named in the message; ``None`` where
        several methods share the code that needs them (the 3D-Nc rate).
    """

    roles: tuple[Role, ...]
    method: Method | None

    def __init__(self, roles: tuple[Role, ...], method: Method | None = None) -> None:
        unknown = ", ".join(roles)
        message = (
            f"both covariances are needed; unknown: {unknown}"
            if method is None
            else (
                f"{method} needs both covariances; unknown: {unknown} "
                "(use method='auto', 'frisbee' or 'alfano' for a maximum Pc)"
            )
        )
        super().__init__(message)
        self.roles = roles
        self.method = method


class UnusableOrbit(GeometryError):
    """An object's orbit has no equinoctial elements (not elliptic,
    rectilinear, or on the elements' singularity), so the 3D-Nc rate is
    undefined.

    Parameters
    ----------
    role
        The object.
    reason
        Why, as :func:`~keepout.equinoctial.to_equinoctial` says; it is the
        message.
    """

    role: Role

    def __init__(self, role: Role, reason: str) -> None:
        super().__init__(reason)
        self.role = role


def require_known_covariance(
    unknown: tuple[Role, ...], method: Method | None = None
) -> None:
    """A check that no covariance is unknown (:class:`MissingCovariance` if one is).

    Parameters
    ----------
    unknown
        Objects whose covariance is unknown.
    method
        Method name, for the message (see :class:`MissingCovariance`).

    Raises
    ------
    MissingCovariance
        If ``unknown`` is not empty.
    """
    if unknown:
        raise MissingCovariance(unknown, method)
