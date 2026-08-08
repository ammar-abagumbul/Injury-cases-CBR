"""
Section-by-section extraction strategy.

The judgment is split into logical sections.
Each section is extracted independently via ``with_structured_output()``
against the corresponding Pydantic sub-model.
Results are merged into the final ``Case``.
"""

from __future__ import annotations

import re
import time
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage

from hklii_psla.extractor.base import BaseExtractor, ExtractionResult
from hklii_psla.extractor.single_pass import _estimate_tokens
from hklii_psla.schemas import (
    ALL_LOSS_CATEGORIES,
    CaseMetadata,
    PlaintiffBackground,
    InjurySummary,
    Treatment,
    LossSummary,
    PSLAAward,
    DeathInfo,
    InjuryLossRelationList,
    Case,
)

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

SECTION_MARKERS = [
    "INTRODUCTION",
    "BACKGROUND",
    "THE ACCIDENT",
    "THE PLAINTIFF'S INJURIES",
    "TREATMENT",
    "MEDICAL",
    "EXPERT EVIDENCE",
    "LOSS",
    "DAMAGES",
    "ASSESSMENT OF DAMAGES",
    "PAIN, SUFFERING AND LOSS OF AMENITIES",
    "PSLA",
    "LOSS OF EARNINGS",
    "PRE-TRIAL LOSS",
    "FUTURE LOSS",
    "SPECIAL DAMAGES",
    "INTEREST",
    "CONCLUSION",
    "ORDER",
    "COSTS",
]

SECTION_PATTERN = re.compile(r"(\n[A-Z][A-Z\s]{3,}(?:\n|$))")


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split judgment into (section_name, section_text) pairs."""
    parts = SECTION_PATTERN.split(text)
    sections: list[tuple[str, str]] = []

    if len(parts) <= 1:
        return [("FULL_JUDGMENT", text)]

    current_name = "PREAMBLE"
    current_text = parts[0]

    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((current_name, current_text))
        current_name = header
        current_text = body
        i += 2

    if current_text.strip():
        sections.append((current_name, current_text))

    return sections


# ---------------------------------------------------------------------------
# Section extractors — each maps a logical section key to a Pydantic model
# and a brief system prompt describing what to extract.
# ---------------------------------------------------------------------------

_SECTION_DEFINITIONS: list[tuple[str, type, str]] = [
    (
        "metadata",
        CaseMetadata,
        "You are a legal citation expert. Your sole task is to identify case "
        "metadata from Hong Kong judgments:\n"
        "- neutral_citation, action_number, case_name\n"
        "- judgment_date in YYYY-MM-DD format\n"
        "- plaintiff_count\n"
        "- has_pre_existing_injuries (true/false)\n"
        "Only report information explicitly stated.",
    ),
    (
        "plaintiff",
        PlaintiffBackground,
        "You are an employment and demographic analyst. Extract plaintiff background:\n"
        "- gender: 'Male' or 'Female'\n"
        "- age_at_accident, age_at_trial\n"
        "- occupation_before, occupation_after, expected_occupation_after\n"
        "- salary_before, salary_after, expected_salary_after (monthly HKD)\n"
        "- education_level\n"
        "Only information explicitly stated. Do not infer.",
    ),
    (
        "injuries",
        InjurySummary,
        "You are a medical-legal analyst. Extract all injuries mentioned in the judgment.\n"
        "For each injury: description, injury_type ('Permanent injury' / "
        "'Temporary injury' / 'Residual disability' / 'Other injury'), body_part.\n"
        "Also determine overall_category: 'Non-serious injury', 'Serious injury', "
        "'Substantial injury', 'Gross disability', or 'Disaster'.\n"
        "Only when explicitly stated.",
    ),
    (
        "treatment",
        Treatment,
        "You are a medical treatment analyst. Extract treatment details:\n"
        "- treatments_received, treatments_future\n"
        "- hospitalisation_days, expected_hospitalisation_days\n"
        "- operations_count, future_operations_count\n"
        "- sick_leave_days_actual, sick_leave_days_expected\n"
        "Only report numbers explicitly stated.",
    ),
    (
        "losses",
        LossSummary,
        "You are a damages and loss assessment specialist. Identify all losses "
        "mentioned in the judgment. Use EXACTLY these loss category names:\n"
        + "\n".join(f"- {c}" for c in ALL_LOSS_CATEGORIES),
    ),
    (
        "psla",
        PSLAAward,
        "You are a PSLA damages specialist. Extract PSLA award details and "
        "comparable cases. For comparable cases extract: case_name, neutral_citation, "
        "action_number, injury_description, hospitalisation_days, operations_count, "
        "sick_leave_days, comparison ('more serious' / 'similar or same' / "
        "'less serious'), psla_amount.\n"
        "Only extract cases explicitly cited in the judgment.",
    ),
    (
        "death",
        DeathInfo,
        "You analyse whether the case involves death. Extract: conscious_before_death, "
        "hours_between_accident_and_death, days_between_accident_and_death.\n"
        "If the case does not involve death, return the default values (false, null, null).",
    ),
    (
        "injury_loss_relations",
        InjuryLossRelationList,
        "You identify explicit links between specific injuries and specific losses "
        "in the judgment. For each link: injury description, loss category, and "
        "evidence (paragraph references or direct quotes).\n"
        "Only extract relationships that are EXPLICITLY stated. Do NOT infer.",
    ),
]


class SectionBySectionExtractor(BaseExtractor):
    """Extract each section independently via ``with_structured_output()``, then merge."""

    def extract(
        self,
        judgment_text: str,
        prompt_style: str = "zero-shot",
    ) -> ExtractionResult:
        model_name = self.model_name or getattr(self.model, "model_name", "unknown")
        t0 = time.perf_counter()
        total_tokens = 0

        combined_text = judgment_text

        results: dict[str, Any] = {}
        errors: list[str] = []

        for section_key, pydantic_model, system_instructions in _SECTION_DEFINITIONS:
            system_prompt = (
                f"{system_instructions}\n\n"
                "Only include information explicitly stated in the text. "
                "Use null / defaults for missing fields."
            )
            user_prompt = (
                f"Extract the {section_key} information from this judgment text.\n\n"
                f"---\n{combined_text[:80000]}\n---"
            )

            try:
                structured = self.model.with_structured_output(pydantic_model)
                extracted = structured.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                results[section_key] = extracted
                total_tokens += _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
            except Exception as e:
                errors.append(f"{section_key}: {e}")

        # Merge section results into a single Case
        merged = _merge_section_results(results)

        duration_ms = (time.perf_counter() - t0) * 1000
        error_str = "; ".join(errors) if errors else None

        try:
            case = Case.model_validate(merged)  # type: ignore[arg-type]
        except Exception as e:
            return ExtractionResult(
                case=None,
                model_name=model_name,
                prompt_style=prompt_style,
                duration_ms=duration_ms,
                token_count=total_tokens,
                error=f"Merge validation failed: {e}",
            )

        return ExtractionResult(
            case=case,
            raw_response=None,
            model_name=model_name,
            prompt_style=prompt_style,
            duration_ms=duration_ms,
            token_count=total_tokens,
            error=error_str if not case else None,
        )


_KEYS: list[str] = [
    "metadata",
    "plaintiff",
    "injuries",
    "treatment",
    "losses",
    "psla",
    "death",
    "injury_loss_relations",
]


def _merge_section_results(parts: dict[str, Any]) -> dict:
    """Merge per-section Pydantic model instances into a single ``Case``-compatible dict."""
    merged: dict = {
        "metadata": {"neutral_citation": "", "action_number": "", "case_name": ""},
        "plaintiff": {},
        "injuries": {"injuries": [], "overall_category": None},
        "treatment": {},
        "losses": {"losses": []},
        "psla": {"amount": None, "comparable_cases": []},
        "death": None,
        "injury_loss_relations": [],
    }

    _merge_from_instance(merged, parts)

    # DeathInfo needs special handling: None means "not applicable".
    # A DeathInfo with all defaults (conscious_before_death=False, etc.) should be None.
    death_val = merged.get("death")
    if death_val is not None and isinstance(death_val, dict) and not death_val.get("conscious_before_death", False):
        merged["death"] = None

    return merged


def _merge_from_instance(merged: dict, parts: dict[str, Any]) -> None:
    """Merge Pydantic model instances from ``parts`` into ``merged``."""
    for key in _KEYS:
        obj = parts.get(key)
        if obj is None:
            continue
        if key == "injury_loss_relations":
            ilr = cast(InjuryLossRelationList, obj)
            merged["injury_loss_relations"] = list(ilr.injury_loss_relations)
            continue
        merged[key] = obj.model_dump()
