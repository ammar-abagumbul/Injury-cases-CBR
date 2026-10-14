"""Injury severity assessment.

Standalone from the ``experiments/`` package on purpose: this is a reusable
feature-engineering step (case -> severity signal), not an evaluative
experiment. It is designed to be consumed by future CBR models (e.g. as a
new weighted attribute alongside ``experiment4``'s ICD/demographic/loss
attributes) without requiring any change to existing experiments.
"""

from hklii_psla.severity.classify import assess_case, assess_case_directory
from hklii_psla.severity.icd_grouping import group_injuries, truncate_code
from hklii_psla.severity.ichi_actions import ICHI_ACTIONS, ActionTier, IchiActionInfo, get_action
from hklii_psla.severity.ichi_taxonomy import IchiEntity, IchiTaxonomy
from hklii_psla.severity.models import (
    IcdGroup,
    InjurySeverityAssessment,
    SeverityConfig,
    TreatmentComplexityScore,
)
from hklii_psla.severity.treatment_complexity import classify_severity, score_treatment_complexity
from hklii_psla.severity.treatment_narrative import (
    TreatmentNarrativeAssessment,
    TreatmentSeverityItem,
    assess_treatment_narrative,
)

__all__ = [
    "assess_case",
    "assess_case_directory",
    "assess_treatment_narrative",
    "classify_severity",
    "group_injuries",
    "get_action",
    "ActionTier",
    "IcdGroup",
    "IchiActionInfo",
    "IchiEntity",
    "IchiTaxonomy",
    "ICHI_ACTIONS",
    "InjurySeverityAssessment",
    "score_treatment_complexity",
    "SeverityConfig",
    "TreatmentComplexityScore",
    "TreatmentNarrativeAssessment",
    "TreatmentSeverityItem",
    "truncate_code",
]
