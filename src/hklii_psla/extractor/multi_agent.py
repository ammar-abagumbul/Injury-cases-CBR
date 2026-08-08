"""
Multi-agent extraction strategy.

Uses separate LLM calls per logical domain (metadata, plaintiff, injuries, etc.),
each acting as a specialised "agent" with domain-specific prompts.
Each agent uses ``with_structured_output(PydanticModel)`` for native schema enforcement.
"""

from __future__ import annotations

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
# Agent definitions — each maps to a Pydantic model + role-specific system prompt
# ---------------------------------------------------------------------------

AGENTS: list[tuple[str, str, type, str]] = [
    (
        "metadata_agent",
        "Case metadata specialist",
        CaseMetadata,
        "You are a legal citation expert. Your sole task is to identify case "
        "metadata from Hong Kong judgments:\n"
        "- neutral_citation, action_number, case_name\n"
        "- judgment_date in YYYY-MM-DD format\n"
        "- plaintiff_count\n"
        "- has_pre_existing_injuries (true/false)\n"
        "Only report information explicitly stated in the text.",
    ),
    (
        "plaintiff_agent",
        "Plaintiff background specialist",
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
        "injury_agent",
        "Medical injury analyst",
        InjurySummary,
        "You are a medical-legal analyst. Extract all injuries mentioned.\n"
        "For each injury: description, injury_type ('Permanent injury' / "
        "'Temporary injury' / 'Residual disability' / 'Other injury'), body_part.\n"
        "Also determine overall_category: 'Non-serious injury', 'Serious injury', "
        "'Substantial injury', 'Gross disability', or 'Disaster'.\n"
        "Only when explicitly stated.",
    ),
    (
        "treatment_agent",
        "Treatment analyst",
        Treatment,
        "You are a medical treatment analyst. Extract treatment details:\n"
        "- treatments_received, treatments_future\n"
        "- hospitalisation_days, expected_hospitalisation_days\n"
        "- operations_count, future_operations_count\n"
        "- sick_leave_days_actual, sick_leave_days_expected\n"
        "Only report numbers explicitly stated.",
    ),
    (
        "loss_agent",
        "Loss assessment specialist",
        LossSummary,
        "You are a damages and loss assessment specialist. Identify all losses "
        "mentioned in the judgment. Use EXACTLY these loss category names:\n"
        + "\n".join(f"- {c}" for c in ALL_LOSS_CATEGORIES),
    ),
    (
        "psla_agent",
        "PSLA award analyst",
        PSLAAward,
        "You are a PSLA damages specialist. Extract PSLA award details and "
        "comparable cases. For comparable cases extract: case_name, neutral_citation, "
        "action_number, injury_description, hospitalisation_days, operations_count, "
        "sick_leave_days, comparison ('more serious' / 'similar or same' / "
        "'less serious'), psla_amount.\n"
        "Only extract cases explicitly cited in the judgment.",
    ),
    (
        "death_agent",
        "Death circumstances analyst",
        DeathInfo,
        "You analyse whether the case involves death. Extract: conscious_before_death, "
        "hours_between_accident_and_death, days_between_accident_and_death.\n"
        "If the case does not involve death, return the default values (false, null, null).",
    ),
    (
        "relation_agent",
        "Injury-loss relation analyst",
        InjuryLossRelationList,
        "You identify explicit links between specific injuries and specific losses "
        "in the judgment. For each link: injury description, loss category, and "
        "evidence (paragraph references or direct quotes).\n"
        "Only extract relationships that are EXPLICITLY stated. Do NOT infer.",
    ),
]


class MultiAgentExtractor(BaseExtractor):
    """Run specialised agents in sequence, each focused on one domain."""

    def extract(
        self,
        judgment_text: str,
        prompt_style: str = "zero-shot",
    ) -> ExtractionResult:
        model_name = self.model_name or getattr(self.model, "model_name", "unknown")
        t0 = time.perf_counter()
        total_tokens = 0

        # Truncate for very long judgments
        text = judgment_text[:120_000]

        results: dict[str, Any] = {}
        errors: list[str] = []

        for agent_name, role, pydantic_model, system_instructions in AGENTS:
            system = (
                f"{system_instructions}\n\n"
                "Only include information explicitly stated in the text. "
                "Use null / defaults for missing fields."
            )
            user = (
                f"Extract the requested information from this judgment.\n\n"
                f"Judgment:\n---\n{text}\n---"
            )

            try:
                structured = self.model.with_structured_output(pydantic_model)
                extracted = structured.invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=user),
                ])
                # Map Pydantic model type to a key name for merging
                model_type_to_key = {
                    CaseMetadata: "metadata",
                    PlaintiffBackground: "plaintiff",
                    InjurySummary: "injuries",
                    Treatment: "treatment",
                    LossSummary: "losses",
                    PSLAAward: "psla",
                    DeathInfo: "death",
                    InjuryLossRelationList: "injury_loss_relations",
                }
                key = model_type_to_key.get(type(pydantic_model), agent_name)  # type: ignore[arg-type]
                results[key] = extracted

                total_tokens += _estimate_tokens(system) + _estimate_tokens(user)
            except Exception as e:
                errors.append(f"{agent_name}: {e}")

        # Merge
        merged = _merge_agent_results(results)

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


def _merge_agent_results(results: dict[str, Any]) -> dict:
    """Merge per-agent Pydantic model instances into a ``Case``-compatible dict."""
    merged: dict = {
        "metadata": _safe_dump(results.get("metadata"), {"neutral_citation": "", "action_number": "", "case_name": ""}),
        "plaintiff": _safe_dump(results.get("plaintiff"), {}),
        "injuries": _safe_dump(results.get("injuries"), {"injuries": [], "overall_category": None}),
        "treatment": _safe_dump(results.get("treatment"), {}),
        "losses": _safe_dump(results.get("losses"), {"losses": []}),
        "psla": _safe_dump(results.get("psla"), {"amount": None, "comparable_cases": []}),
    }

    # Death — None means "not applicable"
    death_data = results.get("death")
    if death_data is not None:
        death_dict: dict[str, Any] = death_data.model_dump()  # type: ignore[union-attr]
        # A DeathInfo with conscious_before_death=False + no specifics → None
        if death_dict.get("conscious_before_death") is False:
            merged["death"] = None
        else:
            merged["death"] = death_dict
    else:
        merged["death"] = None

    # Injury-loss relations
    ilr = cast(InjuryLossRelationList | None, results.get("injury_loss_relations"))
    if ilr is not None:
        merged["injury_loss_relations"] = ilr.injury_loss_relations
    else:
        merged["injury_loss_relations"] = []

    return merged


def _safe_dump(obj: Any, default: dict) -> dict:
    """Convert a Pydantic model to dict, or return ``default``."""
    if obj is None:
        return default
    # All extracted values are Pydantic models with model_dump()
    return obj.model_dump()  # type: ignore[union-attr]
