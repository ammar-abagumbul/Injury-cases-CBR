"""Data types for injury severity assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hklii_psla.schemas import InjuryCategory
from hklii_psla.severity.treatment_narrative import TreatmentNarrativeAssessment


@dataclass
class SeverityConfig:
    """Tunable weights/thresholds for treatment-complexity scoring.

    No numeric-field saturation caps or keyword lists here any more — the
    scoring is driven entirely by ICHI-classified treatment items (see
    ``treatment_narrative.py``/``ichi_actions.py``) plus each injury's own
    ``injury_type`` flag. What's left to tune is how those axes combine.
    """

    # Component weights (need not sum to 1; composite is a weighted average).
    weight_acute_intervention: float = 1.2
    weight_recovery_burden: float = 1.0
    weight_permanence: float = 1.3

    # Composite-score thresholds mapping to InjuryCategory, low -> high.
    # A composite < threshold_serious -> NON_SERIOUS, etc. The top bucket
    # (DISASTER) is intentionally left for manual review rather than an
    # automatic threshold — see ``classify_severity``.
    threshold_serious: float = 0.25
    threshold_substantial: float = 0.5
    threshold_gross_disability: float = 0.70


@dataclass
class TreatmentComplexityScore:
    """Decomposed treatment-complexity signal for one case.

    Each component is in ``[0.0, 1.0]``; ``composite`` is their weighted
    average. ``data_completeness`` is the fraction of classified treatment
    items that resolved to a known ICHI action code (vs. ``NO_MATCH`` or an
    unrecognised code) — 0.0 when there was no narrative assessment at all
    (``llm`` not supplied, or nothing to read). A low value means the score
    rests on little or no real classification; treat it with more caution.
    """

    acute_intervention: float
    recovery_burden: float
    permanence: float
    composite: float
    data_completeness: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "acute_intervention": round(self.acute_intervention, 4),
            "recovery_burden": round(self.recovery_burden, 4),
            "permanence": round(self.permanence, 4),
            "composite": round(self.composite, 4),
            "data_completeness": round(self.data_completeness, 4),
            "notes": self.notes,
        }


@dataclass
class IcdGroup:
    """One injury, mapped to a coarse (non-leaf) ICD-11 group."""

    injury_id: str
    original_code: str | None
    group_code: str | None
    group_description: str | None
    truncated: bool  # True if group_code is shallower than original_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "injury_id": self.injury_id,
            "original_code": self.original_code,
            "group_code": self.group_code,
            "group_description": self.group_description,
            "truncated": self.truncated,
        }


@dataclass
class InjurySeverityAssessment:
    """Full severity assessment for one case."""

    case_id: str
    treatment_complexity: TreatmentComplexityScore
    estimated_category: InjuryCategory
    extracted_category: InjuryCategory | None  # from Case.injuries.overall_category, if present
    agrees_with_extracted: bool | None  # None if extracted_category is absent
    icd_groups: list[IcdGroup]
    narrative: TreatmentNarrativeAssessment | None = None  # LLM reading, if one was requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "treatment_complexity": self.treatment_complexity.to_dict(),
            "estimated_category": self.estimated_category.value,
            "extracted_category": self.extracted_category.value if self.extracted_category else None,
            "agrees_with_extracted": self.agrees_with_extracted,
            "icd_groups": [g.to_dict() for g in self.icd_groups],
            "narrative": self.narrative.model_dump(mode="json") if self.narrative else None,
        }
