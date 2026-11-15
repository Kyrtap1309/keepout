"""Where a Pc comes from: the covariance behind it, as a figure labels it."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from keepout.synthetic.operator import HEADER_COMMENTS

if TYPE_CHECKING:
    from keepout.cdm import Cdm
    from keepout.risk import PcObservation

__all__ = ["Provenance", "covariance_provenance", "provenance"]


class Provenance(enum.StrEnum):
    """What a Pc rests on, in words."""

    CDM_COVARIANCE = "CDM covariance"
    """The covariances the CDM gives, from orbit determination."""
    GP_AGE_MODEL = "assumed (GP age model)"
    """Covariances assumed by keepout's synthetic operator from the age of
    the GP data (:mod:`keepout.synthetic`)."""
    ASSUMED = "assumed covariance"
    """A covariance the CDM marks as assumed (``COVARIANCE_METHOD =
    DEFAULT``), from an unknown model."""
    MAX_PC = "max-Pc (no covariance)"
    """A maximum Pc: an upper bound found without a covariance."""


def provenance(observation: PcObservation, cdm: Cdm) -> Provenance:
    """The provenance of the Pc of a CDM.

    Parameters
    ----------
    observation
        The observation of ``cdm``.
    cdm
        The message; its header comments tell the synthetic operator's
        messages (:data:`keepout.synthetic.operator.HEADER_COMMENTS`) apart.

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
    return _covariance(cdm, assumed=observation.assumed_covariance)


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
    return _covariance(cdm, assumed=cdm.assumed_covariance)


def _covariance(cdm: Cdm, *, assumed: bool) -> Provenance:
    """The provenance of an estimate on ``cdm``'s covariances, ``assumed``
    or not."""
    if not assumed:
        return Provenance.CDM_COVARIANCE
    if set(HEADER_COMMENTS) <= set(cdm.header.comments):
        return Provenance.GP_AGE_MODEL
    return Provenance.ASSUMED
