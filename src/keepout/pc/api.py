"""One entry point for every Pc method, with applicability checks."""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Self,
    TypeAlias,
    TypedDict,
    overload,
)

from keepout._methods import MethodChoice
from keepout._pickling import reduce_with_mappings
from keepout._types import override
from keepout.errors import PcConvergenceError
from keepout.pc._auto import (
    AUTO_MC_SAMPLES,
    AUTO_MC_SEED,
    TCA_OFFSET_LIMIT,
    applicability,
    assess,
    auto,
    auto_options,
    known_only_in_position,
)
from keepout.pc._duration import DURATION_LIMIT, SIGMA_EXTENT, encounter_duration
from keepout.pc.elrod import elrod
from keepout.pc.foster import foster
from keepout.pc.hall3d import hall3d
from keepout.pc.maxpc import alfano, dilution, frisbee
from keepout.pc.montecarlo import montecarlo

if TYPE_CHECKING:
    # typing.Unpack types **kwargs from Python 3.12; typing_extensions stays
    # out of the dependencies.
    from collections.abc import Callable, Mapping

    from typing_extensions import TypeIs, Unpack

    from keepout.conjunction import Conjunction
    from keepout.pc.result import Method, PcResult

__all__ = [
    "AUTO",
    "AUTO_MC_SAMPLES",
    "AUTO_MC_SEED",
    "DURATION_LIMIT",
    "METHODS",
    "SIGMA_EXTENT",
    "TCA_OFFSET_LIMIT",
    "ElrodOptions",
    "FosterOptions",
    "Hall3dOptions",
    "MethodChoice",
    "MethodKind",
    "MethodSpec",
    "MonteCarloOptions",
    "PcComputation",
    "applicability",
    "computation",
    "compute_pc",
    "encounter_duration",
    "require_options",
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
        """The method's result on a conjunction, with options it takes.

        Parameters
        ----------
        conjunction
            The conjunction at TCA.
        options
            Options of the method, by name; each must be in :attr:`options`.

        Returns
        -------
        PcResult
            The method's Pc of ``conjunction``.
        """
        if self.on_plane:
            return self.function(
                conjunction.plane, conjunction.combined_hard_body_radius, **options
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


class FosterOptions(TypedDict, total=False):
    """The options of :func:`~keepout.pc.foster`; every key is optional."""

    rtol: float
    """Relative tolerance of the quadrature (:func:`scipy.integrate.quad`);
    default :data:`~keepout.pc.foster.FOSTER_RTOL` (1e-10), at least
    :data:`~keepout.pc.foster.MIN_RTOL`."""


class ElrodOptions(TypedDict, total=False):
    """The options of :func:`~keepout.pc.elrod`; every key is optional."""

    order: int | None
    """Number of Gauss-Chebyshev nodes, at least 1. ``None`` (default)
    adapts the order from :data:`~keepout.pc.elrod.START_ORDER` up to
    :data:`~keepout.pc.elrod.MAX_ORDER` until the Pc converges; CARA uses
    64."""


class Hall3dOptions(TypedDict, total=False):
    """The options of :func:`~keepout.pc.hall3d`; every key is optional."""

    window: tuple[float, float] | None
    """Integration window ``(start, end)``, in seconds from the
    conjunction's TCA (negative before it), finite with ``start < end``.
    ``None`` (default) grows a window about TCA and the straight-line closest
    approach until the collision rate at its edges is negligible, up to one
    orbital period (the shorter of the two objects') wide."""
    rtol: float
    """Relative tolerance of the time integral of the collision rate;
    default :data:`~keepout.pc.hall3d.HALL_RTOL` (1e-6)."""


class MonteCarloOptions(TypedDict, total=False):
    """The options of :func:`~keepout.pc.montecarlo`; every key is optional."""

    window: tuple[float, float] | None
    """Window ``(start, end)`` the samples are propagated over, in seconds
    from the conjunction's TCA (negative before it), finite with
    ``start < end``. ``None`` (default) sizes it as Hall's method does (see
    :attr:`Hall3dOptions.window`)."""
    samples: int | None
    """Exact number of samples, at least 1. ``None`` (default) samples in
    batches until the confidence interval's half-width is at most ``rtol``
    times the estimate, or ``max_samples`` is reached."""
    rtol: float
    """Target relative half-width of the confidence interval, when
    ``samples`` is ``None``; default :data:`~keepout.pc.montecarlo.MC_RTOL`
    (0.1)."""
    max_samples: int
    """Most samples drawn when stopping on ``rtol``; default
    :data:`~keepout.pc.montecarlo.MC_MAX_SAMPLES` (10 million)."""
    confidence: float
    """Confidence level of the Clopper-Pearson interval, in (0, 1); default
    :data:`~keepout.pc.montecarlo.MC_CONFIDENCE` (0.99)."""
    seed: int | None
    """Seed of the random generator: the same seed gives the same result,
    whatever ``workers``. ``None`` (default): a fresh seed each call."""
    workers: int | None
    """Number of processes to sample in: 1 (default) samples in this
    process, ``None`` uses one per CPU, at most 4. A script that asks for
    more than 1 needs an ``if __name__ == "__main__":`` guard."""


@overload
def compute_pc(
    conjunction: Conjunction,
    method: Literal["auto", "frisbee", "dilution", "alfano"] = ...,
) -> PcResult: ...
@overload
def compute_pc(
    conjunction: Conjunction,
    method: Literal["foster"],
    **options: Unpack[FosterOptions],
) -> PcResult: ...
@overload
def compute_pc(
    conjunction: Conjunction, method: Literal["elrod"], **options: Unpack[ElrodOptions]
) -> PcResult: ...
@overload
def compute_pc(
    conjunction: Conjunction,
    method: Literal["hall3d"],
    **options: Unpack[Hall3dOptions],
) -> PcResult: ...
@overload
def compute_pc(
    conjunction: Conjunction,
    method: Literal["montecarlo"],
    **options: Unpack[MonteCarloOptions],
) -> PcResult: ...


def compute_pc(
    conjunction: Conjunction, method: MethodChoice = "auto", **options: object
) -> PcResult:
    """Probability of collision of a conjunction.

    ``"auto"`` picks ``"hall3d"`` at zero relative velocity or when a 2-D
    assumption is questionable (:func:`keepout.pc.api.applicability`), a maximum Pc
    (``"alfano"``, ``"frisbee"``) when a covariance is unknown, and
    ``"foster"`` otherwise; it falls back to a seeded ``"montecarlo"`` when
    Hall's integral fails. The rules, the fallback and the cost of each
    method are in the guide's "Choosing a method" (``docs/pc-3d.md``).

    Parameters
    ----------
    conjunction
        The conjunction at TCA.
    method
        A method name, or ``"auto"`` (default).
    **options
        The keyword-only parameters of an explicitly chosen method (e.g.
        ``rtol``, ``order``, ``window``, ``samples``, ``seed``), as
        :class:`FosterOptions`, :class:`ElrodOptions`, :class:`Hall3dOptions`
        and :class:`MonteCarloOptions` type them; none with ``"auto"`` or the
        maximum-Pc methods.

    Returns
    -------
    PcResult
        The Pc (``value``) and the ``method`` that computed it, its ``kind``
        (an estimate or an upper bound), a confidence ``interval`` (Monte
        Carlo only), ``diluted`` (dilution only), method-specific
        ``details`` and the ``diagnostics``: the conjunction's own, then the
        applicability warnings (whatever the method), then the method's.

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
        ``"auto"``, if the encounter scan failed, since the Monte Carlo
        fallback shares that scan.
    """
    return _compute(conjunction, method, options)


def _compute(
    conjunction: Conjunction, method: MethodChoice, options: Mapping[str, object]
) -> PcResult:
    """:func:`compute_pc` with its options as a mapping."""
    require_options(method, options)
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


def require_options(method: MethodChoice, options: Mapping[str, object]) -> None:
    """A check that ``method`` is known and takes every option in ``options``.

    Parameters
    ----------
    method
        A method name or ``"auto"``.
    options
        Options to pass to the method, by name.

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
    if not _is_method(method):
        raise ValueError(
            f"unknown Pc method {method!r}; expected 'auto' or one of {tuple(METHODS)}"
        )
    return METHODS[method]


def _is_method(name: str) -> TypeIs[Method]:
    """Whether ``name`` is a method of :data:`METHODS`."""
    return name in METHODS


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
    fallback
        What computes the Pc where this method does not converge, with a
        note saying so; ``None`` (the default): the error is raised.
        :meth:`resolved` sets it as ``"auto"`` falls back from Hall's method.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or options are given with ``"auto"``.
    TypeError
        If the method does not take an option.
    """

    method: MethodChoice = "auto"
    options: Mapping[str, object] = field(default_factory=dict)
    fallback: PcComputation | None = None

    def __post_init__(self) -> None:
        require_options(self.method, self.options)
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
        """The Pc of ``conjunction`` (see :func:`compute_pc`), by the
        fallback where this method does not converge.

        Parameters
        ----------
        conjunction
            The conjunction at TCA.

        Returns
        -------
        PcResult
            The Pc, with a note if the fallback computed it.

        Raises
        ------
        PcConvergenceError
            If the method does not converge and there is no fallback.
        """
        try:
            return _compute(conjunction, self.method, self.options)
        except PcConvergenceError as exc:
            if self.fallback is None:
                raise
            result = self.fallback(conjunction)
            # The reason without the error's advice, which the fallback follows.
            reason = (
                exc.__cause__ if isinstance(exc.__cause__, PcConvergenceError) else exc
            )
            note = f"{self.method} failed ({reason}); {result.method} used instead"
            return dataclasses.replace(result, diagnostics=(*result.diagnostics, note))

    def resolved(self, result: PcResult) -> PcComputation:
        """The computation of the method that gave ``result``, one of this
        computation's: itself if it names that method, else that method as
        ``"auto"`` runs it: Monte Carlo with its seed and sample budget, Hall's
        method with that Monte Carlo as its fallback.

        Parameters
        ----------
        result
            A result of this computation.

        Returns
        -------
        PcComputation
            The computation that reproduces ``result``.

        Raises
        ------
        ValueError
            If this computation names another method than ``result``'s.
        """
        if self.method == result.method:
            return self
        if self.method != "auto":
            raise ValueError(f"a {result.method} result is not one of {self.method}'s")
        if result.method == "hall3d":
            monte_carlo = PcComputation("montecarlo", auto_options("montecarlo"))
            return PcComputation("hall3d", fallback=monte_carlo)
        return PcComputation(result.method, auto_options(result.method))


AUTO: Final = PcComputation()
"""``"auto"`` without options: :func:`compute_pc`'s default."""


@overload
def computation(
    method: Literal["auto", "frisbee", "dilution", "alfano"] = ...,
) -> PcComputation: ...
@overload
def computation(
    method: Literal["foster"], **options: Unpack[FosterOptions]
) -> PcComputation: ...
@overload
def computation(
    method: Literal["elrod"], **options: Unpack[ElrodOptions]
) -> PcComputation: ...
@overload
def computation(
    method: Literal["hall3d"], **options: Unpack[Hall3dOptions]
) -> PcComputation: ...
@overload
def computation(
    method: Literal["montecarlo"], **options: Unpack[MonteCarloOptions]
) -> PcComputation: ...


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

    Returns
    -------
    PcComputation
        The method and its options, callable on a conjunction.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or options are given with ``"auto"``.
    TypeError
        If the method does not take an option.

    Examples
    --------
    ```python
    from keepout import cdm
    from keepout.pc import computation

    foster = computation("foster", rtol=1e-8)
    # A sample CDM in the repository.
    conjunction = cdm.load("tests/cdm/data/tracss_v2.kvn").conjunction(
        combined_hard_body_radius=20.0
    )
    result = foster(conjunction)  # compute_pc(conjunction, "foster", rtol=1e-8)
    print(foster.label, f"{result.value:.3e}")
    # keepout (foster) 4.503e-02
    ```

    Pass it as ``compute=`` to
    :meth:`~keepout.events.ConjunctionEvent.observations`,
    :func:`~keepout.synthetic.run` or
    :func:`~keepout.maneuver.plan_maneuver`.
    """
    return PcComputation(method, options)
