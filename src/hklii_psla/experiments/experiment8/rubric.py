"""Experiment 8 — the severity rubric: schema, prompts, rendering and data-checking of its rules.

A *rubric* describes, in plain language, what cases in each severity tier look like. The LLM
writes it from labelled example cards (``induction``) and may attach a machine-checkable
condition to a claim (``feature operator value``), so claims can be tested against the data.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, create_model
from scipy import stats

Operator = Literal[">=", ">", "<=", "<", "=="]
_OPS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    ">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq,
}

# numeric / count features a rubric rule may refer to (kept to ones a person could check)
CHECKABLE_FEATURES = (
    "n_injuries", "n_distinct_icd_blocks", "n_manifestations", "n_treatments_received",
    "n_surgical_received", "vmis_max_tier_received", "hospitalisation_days", "operations_count",
    "sick_leave_days_actual", "n_losses", "n_loss_categories", "manif_neurological_deficit",
    "injtype_permanent_injury", "age_at_accident", "salary_drop_ratio",
    "injkw_fracture", "injkw_head_brain", "injkw_spinal", "injkw_amputation", "injkw_internal_organ",
    "treatkw_icu",
)


class RubricRule(BaseModel):
    """One claim about a tier, optionally checkable against the extracted data."""

    statement: str = Field(description="The claim in plain words, e.g. 'usually involves two or more operations'.")
    tier: str = Field(description="The tier this claim is typical of.")
    supporting_cases: list[str] = Field(
        default_factory=list, description="Ids (like C012) of the shown cases that support the claim."
    )
    feature: str | None = Field(
        default=None,
        description="If the claim is numeric, one name from the allowed feature list; otherwise null.",
    )
    operator: Operator | None = Field(default=None, description="Comparison for the numeric check, or null.")
    value: float | None = Field(default=None, description="Threshold for the numeric check, or null.")


class TierProfile(BaseModel):
    tier: str
    summary: str = Field(description="Two or three sentences: what a typical case in this tier looks like.")
    typical_features: list[str] = Field(description="Features most cases in the tier share, with typical ranges.")
    versus_lower_tier: str | None = Field(
        default=None, description="What separates this tier from the next lower one; null for the lowest tier."
    )
    boundary_rules: list[RubricRule] = Field(
        default_factory=list, description="If-this-then-at-least-this-tier rules a reader could apply."
    )
    exceptions: list[str] = Field(
        default_factory=list, description="Cases or patterns that do not fit, and when the profile is unreliable."
    )


class Rubric(BaseModel):
    overview: str = Field(description="How severity increases across the tiers, in a short paragraph.")
    profiles: list[TierProfile] = Field(description="One profile per tier, lowest severity first.")
    general_guidance: list[str] = Field(
        default_factory=list, description="Advice for deciding borderline cases, and signs of weak evidence."
    )


def make_prediction_model(tier_names: list[str]) -> type[BaseModel]:
    """Structured output for classifying one case; ``tier`` is restricted to the real tier names."""
    return create_model(
        "Prediction",
        rationale=(str, Field(description="Short reasoning from the case facts.")),
        criteria_used=(list[str], Field(description="The rubric statements or features the decision rests on.")),
        tier=(Literal[tuple(tier_names)], ...),  # type: ignore[valid-type]
        confidence=(Literal["low", "medium", "high"], ...),
    )


# --------------------------------------------------------------------------- prompts

INDUCTION_SYSTEM = (
    "You are an expert in Hong Kong personal injury litigation. Human annotators assigned every "
    "case below to one overall injury-severity tier, listed here from lowest to highest: {tiers}.\n\n"
    "Write a rubric describing what cases in each tier look like, so that a new case could be "
    "classified from it.\n\n"
    "Rules:\n"
    "- Base every claim on the cases shown. Do not rely on general knowledge of what the tier names "
    "mean unless the cases agree.\n"
    "- Prefer features that tell a tier apart from its neighbours over features all tiers share.\n"
    "- Cite supporting case ids (like C012) for each rule.\n"
    "- If a claim is numeric, fill feature / operator / value using ONLY these feature names:\n{features}\n"
    "- Name exceptions: shown cases that contradict a claim.\n"
    "- If a tier has few cases, say its profile is tentative. Never invent thresholds the cases do not show.\n"
    "- Case ids and wording are anonymised; do not guess the court or the year."
)

MERGE_SYSTEM = (
    "You are consolidating several draft rubrics for Hong Kong personal-injury severity tiers "
    "({tiers}, lowest to highest). Each draft was written independently from a different sample of "
    "labelled cases.\n\n"
    "Produce ONE final rubric:\n"
    "- Keep claims that recur across drafts; drop claims that appear in only one draft unless they "
    "are strongly supported.\n"
    "- Merge duplicate wording; where drafts disagree on a threshold, use the middle value and note "
    "the disagreement under exceptions.\n"
    "- Keep the numeric checks (feature / operator / value) of the claims you keep, using only these "
    "feature names:\n{features}\n"
    "- Union the supporting case ids of merged claims (at most 8 per rule).\n"
    "- Keep one profile per tier, lowest first, and stay faithful to the drafts: add nothing new."
)

CLASSIFY_SYSTEM = (
    "You classify Hong Kong personal-injury cases into an overall injury-severity tier chosen by "
    "human annotators. The tiers, lowest to highest: {tiers}.\n"
    "Judge the severity of the injury and its consequences from the case facts shown. "
    "Do not use the claimed compensation amount, which is not shown. {guidance}"
)

RUBRIC_GUIDANCE = (
    "Use the rubric below, which was written from labelled examples. Apply it to the case: say which "
    "rubric statements your decision rests on, and prefer the rubric over your own prior views.\n\n"
    "RUBRIC\n{rubric}"
)


def _features_block(descriptions: dict[str, str]) -> str:
    return "\n".join(f"- {name}: {text}" for name, text in descriptions.items())


def induction_system(tier_names: list[str], descriptions: dict[str, str]) -> str:
    return INDUCTION_SYSTEM.format(tiers=" < ".join(tier_names), features=_features_block(descriptions))


def merge_system(tier_names: list[str], descriptions: dict[str, str]) -> str:
    return MERGE_SYSTEM.format(tiers=" < ".join(tier_names), features=_features_block(descriptions))


def induction_user(cards_by_tier: dict[str, list[str]]) -> str:
    parts = []
    for tier, cards in cards_by_tier.items():
        parts.append(f"=== TIER: {tier} ({len(cards)} cases shown) ===\n\n" + "\n\n".join(cards))
    return "\n\n".join(parts) + "\n\nWrite the rubric."


def merge_user(drafts: list[Rubric]) -> str:
    body = "\n\n".join(f"--- DRAFT {i + 1} ---\n{d.model_dump_json(indent=1)}" for i, d in enumerate(drafts))
    return body + "\n\nWrite the final consolidated rubric."


def classify_system(tier_names: list[str], rubric: Rubric | None) -> str:
    guidance = RUBRIC_GUIDANCE.format(rubric=render_rubric(rubric, with_cases=False)) if rubric else ""
    return CLASSIFY_SYSTEM.format(tiers=" < ".join(tier_names), guidance=guidance)


def classify_user(card: str, examples: list[tuple[str, str]] | None = None) -> str:
    shown = ""
    if examples:
        shown = "Labelled examples of similar cases:\n\n" + "\n\n".join(
            f"{c}\n=> HUMAN TIER: {t}" for c, t in examples
        ) + "\n\n---\n\n"
    return f"{shown}Classify this case:\n\n{card}"


# ------------------------------------------------------------------------- rendering


def render_rubric(rubric: Rubric, with_cases: bool = True) -> str:
    out = [rubric.overview.strip(), ""]
    for p in rubric.profiles:
        out += [f"## {p.tier}", p.summary.strip(), ""]
        if p.typical_features:
            out += ["Typical features:", *[f"- {f}" for f in p.typical_features], ""]
        if p.versus_lower_tier:
            out += [f"Versus the lower tier: {p.versus_lower_tier.strip()}", ""]
        if p.boundary_rules:
            out.append("Rules:")
            for r in p.boundary_rules:
                check = f" [{r.feature} {r.operator} {r.value:g}]" if r.feature and r.operator and r.value is not None else ""
                cases = f" (cases: {', '.join(r.supporting_cases)})" if with_cases and r.supporting_cases else ""
                out.append(f"- {r.statement}{check}{cases}")
            out.append("")
        if p.exceptions:
            out += ["Exceptions:", *[f"- {e}" for e in p.exceptions], ""]
    if rubric.general_guidance:
        out += ["## General guidance", *[f"- {g}" for g in rubric.general_guidance], ""]
    return "\n".join(out)


# ------------------------------------------------------------------------ rule checks


def check_rules(
    rubric: Rubric, X: pd.DataFrame, tiers: pd.Series, mask: np.ndarray, tier_names: list[str]
) -> pd.DataFrame:
    """Test every checkable rule on the cases selected by ``mask``.

    For a rule about tier T, this compares the share of T's cases that satisfy the condition with
    the share of all other cases that do, so a rule is "supported" only if it is really more common
    in T. Cases where the feature is not stated count as not satisfying it.
    """
    rows = []
    t = tiers[mask].reset_index(drop=True)
    for profile in rubric.profiles:
        for rule in profile.boundary_rules:
            if not (rule.feature and rule.operator and rule.value is not None):
                continue
            base = {"tier": rule.tier, "statement": rule.statement,
                    "condition": f"{rule.feature} {rule.operator} {rule.value:g}"}
            if rule.feature not in X.columns or rule.tier not in tier_names:
                rows.append({**base, "verdict": "cannot check (unknown feature or tier)"})
                continue
            ok = _OPS[rule.operator](X.loc[mask, rule.feature].reset_index(drop=True), rule.value).fillna(False)
            in_t = (t == rule.tier).to_numpy()
            a, b = int(ok[in_t].sum()), int(in_t.sum())
            c, d = int(ok[~in_t].sum()), int((~in_t).sum())
            if b == 0 or d == 0:
                rows.append({**base, "verdict": "cannot check (tier absent from these cases)"})
                continue
            p = float(stats.fisher_exact([[a, b - a], [c, d - c]], alternative="greater")[1])
            rows.append({
                **base, "in_tier": f"{a}/{b}", "in_other_tiers": f"{c}/{d}",
                "share_in_tier": a / b, "share_in_others": c / d, "p_value": p,
                "verdict": "supported" if a / b > c / d and p < 0.05 else
                           "weak" if a / b > c / d else "not supported",
            })
    return pd.DataFrame(rows)
