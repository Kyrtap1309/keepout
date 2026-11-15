"""What a Pc rests on: the covariance behind it, or none.

A Pc is an estimate on covariances or an upper bound found without one
(a maximum Pc). The covariances are the CDM's own, from orbit
determination, unless the CDM marks them as assumed
(``COVARIANCE_METHOD = DEFAULT``); keepout's synthetic operator
(:mod:`keepout.synthetic`) marks its assumed covariances with its header
comments (:data:`HEADER_COMMENTS`). Only an assumed covariance needs the
CDM's header comments: :func:`provenance` asks for them only then, so a
caller holding a CDM's text parses it only then.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from keepout.cdm import Cdm
    from keepout.risk.observation import PcObservation

__all__ = ["HEADER_COMMENTS", "Provenance", "covariance_provenance", "provenance"]

HEADER_COMMENTS: Final = (
    "keepout synthetic operator: screening of public GP data with SGP4",
    "Covariances are assumed (a model, not orbit determination)",
)
"""Header comments of the synthetic operator's CDMs."""


class Provenance(enum.StrEnum):
    """What a Pc rests on: the CDM's covariances (``CDM_COVARIANCE``),
    covariances the synthetic operator assumed from the age of the GP data
    (``GP_AGE_MODEL``), other assumed covariances (``ASSUMED``), or none, a
    maximum Pc (``MAX_PC``).

    The values are identifiers, not words to show:
    :data:`keepout.figures.PROVENANCE_LABELS` names each and
    :data:`keepout.figures.PROVENANCE_HELP` explains it.
    """

    CDM_COVARIANCE = "cdm_covariance"
    GP_AGE_MODEL = "gp_age_model"
    ASSUMED = "assumed"
    MAX_PC = "max_pc"


def provenance(
    observation: PcObservation, comments: Callable[[], Iterable[str]]
) -> Provenance:
    """The provenance of the Pc of a CDM.

    Parameters
    ----------
    observation
        The Pc observation of the CDM.
    comments
        The CDM's header comments, called only when ``observation`` is an
        estimate on an assumed covariance: they tell the synthetic
        operator's messages (:data:`HEADER_COMMENTS`) apart. A caller
        holding the CDM's text parses it here, only when needed.

    Returns
    -------
    Provenance
        :attr:`Provenance.MAX_PC` for an upper bound; for an assumed
        covariance :attr:`Provenance.GP_AGE_MODEL` when the header carries
        every comment of the synthetic operator, else
        :attr:`Provenance.ASSUMED`; otherwise
        :attr:`Provenance.CDM_COVARIANCE`.
    """
    if observation.kind == "upper_bound":
        return Provenance.MAX_PC
    if not observation.assumed_covariance:
        return Provenance.CDM_COVARIANCE
    return _assumed(comments())


def covariance_provenance(cdm: Cdm) -> Provenance:
    """The provenance of a Pc on a CDM's covariances, as those of a
    manoeuvre plan made from it (an estimate, never an upper bound).

    Parameters
    ----------
    cdm
        The message; its ``COVARIANCE_METHOD`` says whether its covariances
        are assumed, and its header comments tell the synthetic operator's
        messages apart.

    Returns
    -------
    Provenance
        :attr:`Provenance.GP_AGE_MODEL` or :attr:`Provenance.ASSUMED` for an
        assumed covariance, as :func:`provenance`; otherwise
        :attr:`Provenance.CDM_COVARIANCE`.
    """
    if not cdm.assumed_covariance:
        return Provenance.CDM_COVARIANCE
    return _assumed(cdm.header.comments)


def _assumed(comments: Iterable[str]) -> Provenance:
    """The provenance of an assumed covariance, from its CDM's header
    comments."""
    if set(HEADER_COMMENTS) <= set(comments):
        return Provenance.GP_AGE_MODEL
    return Provenance.ASSUMED
