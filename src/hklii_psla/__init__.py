"""HKLII PSLA Experiments — Structured feature extraction for personal injury cases."""

from hklii_psla.extractor.base import ExtractionMetadata, TokenCount
from hklii_psla.schemas import (
    Case,
    CaseMetadata,
    DeathInfo,
    Injury,
    InjuryLossRelation,
    InjuryLossRelationList,
    InjurySummary,
    Loss,
    LossSummary,
    PlaintiffBackground,
    PSLAAward,
    PSLAComparableCase,
    Treatment,
)

__all__ = [
    "Case",
    "CaseMetadata",
    "DeathInfo",
    "ExtractionMetadata",
    "Injury",
    "InjuryLossRelation",
    "InjuryLossRelationList",
    "InjurySummary",
    "Loss",
    "LossSummary",
    "PlaintiffBackground",
    "PSLAAward",
    "PSLAComparableCase",
    "TokenCount",
    "Treatment",
]
