"""Risk of a conjunction event: risk levels, Pc trend and a recommended action.

The assessment works on :class:`PcObservation` sequences, whatever their
source: keepout's own Pc of each CDM of an event, or Pc values reported by
others (ESA's Kelvins data). :func:`provenance` tells what a Pc rests on
(:class:`Provenance`), :func:`covariance_provenance` what a Pc on a CDM's
covariances does.
"""

from keepout.risk.decision import (
    DEFAULT_DECISION_CONFIG,
    Action,
    Assessment,
    Decision,
    DecisionConfig,
    assess,
    decide,
)
from keepout.risk.observation import PC_FLOOR, PcKind, PcObservation
from keepout.risk.provenance import Provenance, covariance_provenance, provenance
from keepout.risk.thresholds import DEFAULT_THRESHOLDS, Basis, RiskLevel, Thresholds
from keepout.risk.trend import (
    DEFAULT_TREND_CONFIG,
    Direction,
    Trend,
    TrendConfig,
    TrendPoint,
    analyse_trend,
)

__all__ = [
    "DEFAULT_DECISION_CONFIG",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_TREND_CONFIG",
    "PC_FLOOR",
    "Action",
    "Assessment",
    "Basis",
    "Decision",
    "DecisionConfig",
    "Direction",
    "PcKind",
    "PcObservation",
    "Provenance",
    "RiskLevel",
    "Thresholds",
    "Trend",
    "TrendConfig",
    "TrendPoint",
    "analyse_trend",
    "assess",
    "covariance_provenance",
    "decide",
    "provenance",
]
