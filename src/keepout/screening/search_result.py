"""What an :class:`~keepout.screening.tca.ApproachSearch` finds: the minima of
the range inside the volumes and the objects that failed."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from keepout._linalg import as_vector
from keepout._types import FloatArray

__all__ = ["ObjectFailure", "RangeMinimum", "SearchResult"]


@dataclass(frozen=True, order=True, slots=True)
class ObjectFailure:
    """The earliest failed state of an object seen by a search; failures
    are ordered by offset, then code.

    Parameters
    ----------
    offset
        Time of the state [s since the start of the window].
    code
        Its error code (:data:`keepout.propagation.ERROR_MESSAGES`).
    """

    offset: float
    code: int


@dataclass(frozen=True, eq=False, slots=True)
class RangeMinimum:
    """A local minimum of a pair's range, inside the primary's volume.

    Parameters
    ----------
    primary_index
        Position of the primary among the search's primaries.
    secondary_index
        Position of the other object in the search's catalog.
    offset
        Time of closest approach [s since the start of the window].
    range_at_tca
        Range at TCA [m].
    r_primary, v_primary
        The primary's position [m] and velocity [m/s] at TCA, in the
        propagators' frame; read-only copies, shape ``(3,)``.
    r_secondary, v_secondary
        The other object's state at TCA, likewise.

    Raises
    ------
    ValueError
        If a state vector does not have shape ``(3,)`` or is not finite.
    """

    primary_index: int
    secondary_index: int
    offset: float
    range_at_tca: float
    r_primary: FloatArray
    v_primary: FloatArray
    r_secondary: FloatArray
    v_secondary: FloatArray

    def __post_init__(self) -> None:
        for name in ("r_primary", "v_primary", "r_secondary", "v_secondary"):
            object.__setattr__(self, name, as_vector(getattr(self, name), name))


def _minimum_sort_key(minimum: RangeMinimum) -> tuple[int, int, float]:
    """The order of minima in a result: by primary, secondary, then offset."""
    return minimum.primary_index, minimum.secondary_index, minimum.offset


def keep_earliest(
    failures: dict[int, ObjectFailure], index: int, failure: ObjectFailure
) -> None:
    """An object's new failure, stored in ``failures`` in place of the one
    recorded unless that one is earlier (by offset, then code)."""
    known = failures.get(index)
    if known is None or failure < known:
        failures[index] = failure


@dataclass(frozen=True, eq=False, slots=True)
class SearchResult:
    """What a search finds.

    Parameters
    ----------
    minima
        The minima inside the volumes; kept sorted by primary, secondary and
        offset.
    primary_failures, catalog_failures
        The earliest failed state seen of each primary and catalog object,
        by position (copies, sorted by position). Screening skips these
        objects: minima of a failed object are still listed here, from the
        intervals in which both objects were propagated.
    """

    minima: tuple[RangeMinimum, ...]
    primary_failures: Mapping[int, ObjectFailure]
    catalog_failures: Mapping[int, ObjectFailure]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "minima", tuple(sorted(self.minima, key=_minimum_sort_key))
        )
        primary = dict(sorted(self.primary_failures.items()))
        catalog = dict(sorted(self.catalog_failures.items()))
        object.__setattr__(self, "primary_failures", primary)
        object.__setattr__(self, "catalog_failures", catalog)

    @classmethod
    def empty(cls) -> SearchResult:
        """A result with no minima and no failures."""
        return cls((), {}, {})

    @classmethod
    def merge(cls, results: Iterable[SearchResult]) -> SearchResult:
        """One result from the results of searches over parts of a catalog.

        The minima are put together and each object keeps its earliest
        failure; the results must number the objects alike.
        """
        minima: list[RangeMinimum] = []
        primary: dict[int, ObjectFailure] = {}
        catalog: dict[int, ObjectFailure] = {}
        for result in results:
            minima.extend(result.minima)
            for index, failure in result.primary_failures.items():
                keep_earliest(primary, index, failure)
            for index, failure in result.catalog_failures.items():
                keep_earliest(catalog, index, failure)
        return cls(tuple(minima), primary, catalog)
