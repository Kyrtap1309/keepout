"""One entry point for every Pc method, with applicability checks."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, Self, TypeAlias, cast, overload

from keepout._methods import MethodChoice
from keepout._pickling import reduce_with_mappings
from keepout._types import override
from keepout.conjunction import Conjunction
from keepout.pc._auto import (
    AUTO_MC_SAMPLES,
    AUTO_MC_SEED,
    TCA_OFFSET_LIMIT,
    applicability,
    assess,
    auto,
    known_only_in_position,
)
from keepout.pc._duration import DURATION_LIMIT, SIGMA_EXTENT, encounter_duration
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.hall3d import hall3d
from keepout.pc.maxpc import alfano, dilution, frisbee
from keepout.pc.montecarlo import montecarlo
from keepout.pc.result import Method, PcResult

__all__ = [
    "AUTO",
    "AUTO_MC_SAMPLES",
    "AUTO_MC_SEED",
    "DURATION_LIMIT",
    "METHODS",
    "SIGMA_EXTENT",
    "TCA_OFFSET_LIMIT",
    "MethodChoice",
    "MethodKind",
    "MethodSpec",
    "PcComputation",
    "applicability",
    "computation",
    "compute_pc",
    "encounter_duration",
]

MethodKind: TypeAlias = Literal["2d", "3d", "montecarlo", "maxpc"]
"""A family of Pc methods: 2-D on the encounter plane, 3-D, Monte Carlo, or
maximum Pc (upper bounds on the encounter plane)."""


@dataclass(frozen=True, eq=False, slots=True)
class MethodSpec:
    """What a Pc method is and how :func:`compute_pc` runs it.

    Parameters
    ----------
    function
        The method's function: ``f(plane, hard_body_radius, **options)`` for
        a 2-D or maximum-Pc method, ``f(conjunction, **options)`` for a 3-D
        or Monte Carlo one.
    kind
        The method's family.

    Attributes
    ----------
    options
        The keyword-only parameters of ``function``: the options
        :func:`compute_pc` passes on.
    """

    function: Callable[..., PcResult]
    kind: MethodKind
    options: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        parameters = inspect.signature(self.function).parameters.values()
        options = frozenset(
            p.name for p in parameters if p.kind is inspect.Parameter.KEYWORD_ONLY
        )
        object.__setattr__(self, "options", options)

    @property
    def on_plane(self) -> bool:
        """Whether the method works on the encounter plane (2-D, maximum Pc)."""
        return self.kind in ("2d", "maxpc")

    @property
    def uses_velocity(self) -> bool:
        """Whether the method uses the velocity uncertainty (3-D, Monte Carlo)."""
        return not self.on_plane

    def run(self, conjunction: Conjunction, options: Mapping[str, object]) -> PcResult:
        """The method's result on a conjunction, with options it takes."""
        if self.on_plane:
            return self.function(
                conjunction.plane, conjunction.hard_body_radius, **options
            )
        return self.function(conjunction, **options)


METHODS: Final[Mapping[Method, MethodSpec]] = MappingProxyType(
    {
        "foster": MethodSpec(foster, "2d"),
        "elrod": MethodSpec(elrod, "2d"),
        "frisbee": MethodSpec(frisbee, "maxpc"),
        "dilution": MethodSpec(dilution, "maxpc"),
        "alfano": MethodSpec(alfano, "maxpc"),
        "hall3d": MethodSpec(hall3d, "3d"),
        "montecarlo": MethodSpec(montecarlo, "montecarlo"),
    }
)
"""Every Pc method by name: the one table of what each is and takes."""


@overload
def compute_pc(conjunction: Conjunction, method: Literal["auto"] = ...) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction, method: Literal["foster"], *, rtol: float = ...
) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction, method: Literal["elrod"], *, order: int | None = ...
) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction, method: Literal["frisbee", "dilution", "alfano"]
) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction,
    method: Literal["hall3d"],
    *,
    window: tuple[float, float] | None = ...,
    rtol: float = ...,
) -> PcResult: ...


@overload
def compute_pc(
    conjunction: Conjunction,
    method: Literal["montecarlo"],
    *,
    window: tuple[float, float] | None = ...,
    samples: int | None = ...,
    rtol: float = ...,
    max_samples: int = ...,
    confidence: float = ...,
    seed: int | None = ...,
) -> PcResult: ...


def compute_pc(
    conjunction: Conjunction, method: MethodChoice = "auto", **options: object
) -> PcResult:
    """Probability of collision of a conjunction.

    Parameters
    ----------
    conjunction
        The conjunction at TCA.
    method
        A method name, or ``"auto"`` (default), which picks:

        1. ``"hall3d"`` if the relative velocity is zero (both covariances
           must then be known and both orbits elliptic);
        2. ``"alfano"`` if both covariances are unknown, ``"frisbee"`` if one
           is (maximum Pc: upper bounds);
        3. ``"hall3d"`` if a 2-D assumption is questionable (see
           :func:`applicability`), both orbits are elliptic and both
           covariances have a velocity block, with the 2-D Pc kept in
           ``details["pc_2d"]``;
        4. ``"foster"`` otherwise, with a note when a 2-D assumption is
           questionable but a covariance is known only in position (zero
           velocity block, e.g. a synthetic covariance; see
           :attr:`~keepout.conjunction.Conjunction.position_only_covariance`):
           the 3-D methods would take its velocity as exact.

        However chosen, ``"hall3d"`` and ``"montecarlo"`` note a covariance
        known only in position: they take its velocity as exact.

        When ``"hall3d"`` raises :class:`PcConvergenceError` after its
        encounter scan (the peak-overlap iteration failed where the rate
        matters, the rate cannot be computed at a quadrature node, or Nc is
        not finite), ``"auto"`` falls back to ``"montecarlo"`` with at most
        ``AUTO_MC_SAMPLES`` trials and the seed ``AUTO_MC_SEED``, so
        ``"auto"`` is reproducible. A failure of the encounter scan itself
        is not covered: Monte Carlo reuses that scan, so its error
        propagates.
        The 2-D and maximum-Pc methods take milliseconds; ``"hall3d"`` about
        1-60 s and the Monte Carlo fallback up to about a minute.
    **options
        The keyword-only parameters of an explicitly chosen method (e.g.
        ``rtol``, ``order``, ``window``, ``samples``, ``seed``), as typed by
        the overloads; none with ``"auto"`` or the maximum-Pc methods.

    Returns
    -------
    PcResult
        ``diagnostics`` start with the conjunction's own (covariance repairs,
        unknown covariances), then applicability warnings, then the method's.
        Applicability warnings are added, whatever the method, whenever the
        encounter plane is defined (non-zero relative velocity) and they
        fire, also with a covariance unknown (maximum Pc): the
        closest-approach offset does not depend on the covariances, and the
        duration from the known covariance alone is a lower bound, so its
        warning still holds. ``details["encounter_duration_s"]`` is added
        only when both covariances are known.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or options are given with ``"auto"``.
    TypeError
        If an option is not accepted by the method.
    CovarianceError
        If the chosen method needs a covariance that is unknown.
    GeometryError
        If the chosen method needs a plane or orbit that is undefined; with
        ``"auto"``, if the relative velocity is zero and a covariance is
        unknown or an orbit is not elliptic.
    PcConvergenceError
        If ``"hall3d"`` was chosen explicitly and did not converge; if
        ``"montecarlo"`` was chosen and its encounter scan failed; with
        ``"auto"``, if the encounter scan failed (its encounters need more
        detection points than allowed, the rate is significant where the
        screen ruled it negligible, or the rate cannot be computed on the scan
        grid), since the Monte Carlo fallback shares that scan.
    """
    return _compute(conjunction, method, options)


def _compute(
    conjunction: Conjunction, method: MethodChoice, options: Mapping[str, object]
) -> PcResult:
    """:func:`compute_pc` with its options as a mapping."""
    _require_options(method, options)
    if method == "auto":
        result, (notes, details) = auto(conjunction)
    else:
        notes, details = assess(conjunction)
        result = METHODS[method].run(conjunction, options)
    notes += _velocity_notes(conjunction, result.method)
    return dataclasses.replace(
        result,
        diagnostics=conjunction.diagnostics + notes + result.diagnostics,
        details={**result.details, **details},
    )


def _velocity_notes(conjunction: Conjunction, method: Method) -> tuple[str, ...]:
    """A note when ``method`` needs the velocity uncertainty and a covariance
    has none: whichever way the method was chosen, it takes that velocity as
    exact."""
    position_only = conjunction.position_only_covariance
    if not METHODS[method].uses_velocity or not position_only:
        return ()
    return (
        (
            f"{known_only_in_position(conjunction, position_only)}: {method} "
            "takes its velocity as exact"
        ),
    )


def _require_options(method: MethodChoice, options: Mapping[str, object]) -> None:
    """A check that ``method`` is known and takes every option in ``options``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or options are given with ``"auto"``.
    TypeError
        If the method does not take an option.
    """
    if method == "auto":
        if options:
            raise ValueError(f"options {sorted(options)} need an explicit method")
        return
    spec = _spec(method)
    refused = sorted(options.keys() - spec.options)
    if refused:
        takes = ", ".join(sorted(spec.options)) or "none"
        raise TypeError(f"{method} takes no option {refused}; it takes: {takes}")


def _spec(method: str) -> MethodSpec:
    """The :class:`MethodSpec` of a method name.

    Raises
    ------
    ValueError
        If ``method`` names no method (nor does ``"auto"``: it is a choice).
    """
    # A name that is no method is only looked up: it finds nothing.
    spec = METHODS.get(cast(Method, method))
    if spec is None:
        raise ValueError(
            f"unknown Pc method {method!r}; expected 'auto' or one of {tuple(METHODS)}"
        )
    return spec


@dataclass(frozen=True, eq=False, slots=True)
class PcComputation:
    """A Pc method with its options, to run on conjunctions; see
    :func:`computation`.

    Calling it on a conjunction is :func:`compute_pc` with this method and
    these options.

    Parameters
    ----------
    method
        A method name, or ``"auto"``.
    options
        The method's options, as :func:`compute_pc` takes them; stored as a
        read-only copy.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or options are given with ``"auto"``.
    TypeError
        If the method does not take an option.
    """

    method: MethodChoice = "auto"
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_options(self.method, self.options)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[object, ...]]:
        return reduce_with_mappings(self)

    @property
    def label(self) -> str:
        """Who computes the Pc: ``"keepout (<method>)"``.

        A risk observation's source: keepout's results are labelled by the
        method that gave the value (``"keepout (foster)"`` when ``"auto"``
        picked Foster), and failures by the method asked for (``"keepout
        (auto)"``).
        """
        return f"keepout ({self.method})"

    def __call__(self, conjunction: Conjunction) -> PcResult:
        """The Pc of ``conjunction`` (see :func:`compute_pc`)."""
        return _compute(conjunction, self.method, self.options)


AUTO: Final = PcComputation()
"""``"auto"`` without options: :func:`compute_pc`'s default."""


def computation(method: MethodChoice = "auto", **options: object) -> PcComputation:
    """A Pc method with its options, to run on many conjunctions.

    E.g. ``computation("montecarlo", samples=100_000, seed=1)``; the options
    are checked here, not on the first conjunction.

    Parameters
    ----------
    method
        A method name, or ``"auto"`` (default); see :func:`compute_pc`.
    **options
        The method's options (see :func:`compute_pc`).

    Raises
    ------
    ValueError
        If ``method`` is unknown, or options are given with ``"auto"``.
    TypeError
        If the method does not take an option.
    """
    return PcComputation(method, options)
