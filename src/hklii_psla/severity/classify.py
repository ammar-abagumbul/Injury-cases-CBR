"""Top-level orchestration: Case -> InjurySeverityAssessment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from hklii_psla.experiments.experiment4.icd_similarity import IcdTaxonomy
from hklii_psla.schemas import Case
from hklii_psla.severity.icd_grouping import DEFAULT_MAX_DEPTH, group_injuries
from hklii_psla.severity.models import InjurySeverityAssessment, SeverityConfig
from hklii_psla.severity.treatment_complexity import classify_severity, score_treatment_complexity
from hklii_psla.severity.treatment_narrative import assess_treatment_narrative

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def _case_id(case: Case) -> str:
    return (
        case.metadata.neutral_citation
        or case.metadata.action_number
        or case.metadata.case_name
        or "unknown"
    )


def assess_case(
    case: Case,
    taxonomy: IcdTaxonomy,
    config: SeverityConfig | None = None,
    icd_max_depth: int = DEFAULT_MAX_DEPTH,
    llm: BaseChatModel | None = None,
) -> InjurySeverityAssessment:
    """Assess one already-extracted ``Case``: treatment-complexity severity
    plus coarse ICD-11 grouping per injury.

    Severity now depends on ``llm`` classifying the free-text treatment
    narrative into ICHI action codes (see ``treatment_narrative.py``) — it
    is no longer a rule-based fallback, it's the primary signal. Without
    ``llm``, the only remaining input is each injury's own ``injury_type``
    flag: ``acute_intervention``/``recovery_burden`` will be 0.0, meaning
    "not assessed", not "confirmed non-serious". Check
    ``treatment_complexity.data_completeness``/``.notes`` before trusting a
    low composite.
    """
    config = config or SeverityConfig()

    if llm is None:
        print(
            "WARNING: assess_case called without llm — severity will rest only on "
            "injury_type; acute_intervention/recovery_burden are unassessed, not confirmed low.",
            file=sys.stderr,
        )

    narrative = assess_treatment_narrative(llm, case.treatment) if llm is not None else None

    complexity = score_treatment_complexity(case.injuries.injuries, config, narrative)
    llm_category = narrative.overall_severity_category if narrative else None
    estimated_category = classify_severity(complexity.composite, config, llm_category)

    extracted_category = case.injuries.overall_category
    agrees = None if extracted_category is None else (extracted_category == estimated_category)

    icd_groups = group_injuries(case.injuries.injuries, taxonomy, icd_max_depth)

    return InjurySeverityAssessment(
        case_id=_case_id(case),
        treatment_complexity=complexity,
        estimated_category=estimated_category,
        extracted_category=extracted_category,
        agrees_with_extracted=agrees,
        icd_groups=icd_groups,
        narrative=narrative,
    )


def assess_case_directory(
    directory: str | Path,
    taxonomy: IcdTaxonomy | None = None,
    config: SeverityConfig | None = None,
    pattern: str = "*.json",
    icd_max_depth: int = DEFAULT_MAX_DEPTH,
    llm: BaseChatModel | None = None,
) -> list[InjurySeverityAssessment]:
    """Assess every extracted Case JSON file in ``directory``.

    Each file must be a full ``Case`` dump (as produced by any of this
    project's extractors, e.g. ``result.case.model_dump(mode="json")``) —
    not the flattened ``CaseFeatures`` shape ``experiment4`` uses.

    ``llm`` is now effectively required for a meaningful assessment (one
    call per case, for the ICHI narrative classification) — see
    ``assess_case``'s docstring for what you get without it.
    """
    taxonomy = taxonomy or IcdTaxonomy.from_json()
    config = config or SeverityConfig()

    assessments: list[InjurySeverityAssessment] = []
    for path in sorted(Path(directory).glob(pattern)):
        with path.open(encoding="utf-8") as f:
            case = Case.model_validate(json.load(f))
        assessments.append(assess_case(case, taxonomy, config, icd_max_depth, llm))
    return assessments
