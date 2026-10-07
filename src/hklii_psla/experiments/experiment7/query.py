"""Judge-facing feature queries for Experiment 7.

A :class:`FeatureQuery` is the small, human-authored description of a case a
judge wants precedents for.  Injuries may be given either as ICD-11 codes or as
free text (resolved against the ICD-11 descriptions); losses must be canonical
loss-category names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from hklii_psla.experiments.experiment4.icd_similarity import IcdTaxonomy
from hklii_psla.experiments.experiment7.features import CorpusCase
from hklii_psla.schemas import ALL_LOSS_CATEGORIES, InjuryCategory

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "of",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "in",
    "on",
    "with",
    "injury",
    "injuries",
    "fracture",
    "fractures",
    "left",
    "right",
    "bilateral",
}


class FeatureQuery(BaseModel):
    """A judge-supplied feature vector to retrieve similar cases for."""

    query_id: str = "query"
    gender: str | None = None
    age_at_accident: int | None = None
    age_at_trial: int | None = None
    injuries: list[str] = Field(
        default_factory=list,
        description="ICD-11 codes for the plaintiff's injuries.",
    )
    injury_descriptions: list[str] = Field(
        default_factory=list,
        description="Free-text injury descriptions; resolved to ICD-11 codes.",
    )
    losses: list[str] = Field(
        default_factory=list,
        description="Canonical loss-category names.",
    )
    overall_category: str | None = Field(
        default=None,
        description="Overall injury severity category (see InjuryCategory).",
    )
    operations_count: int | None = None
    hospitalisation_days: int | None = None
    psla_target_real: float | None = Field(
        default=None,
        description="Optional expected PSLA in base-year HKD, used for re-ranking.",
    )
    psla_band: list[float] | None = Field(
        default=None,
        description="Optional [min, max] real-HKD window restricting candidates.",
    )
    weights: dict[str, float] | None = None
    k: int = 10

    @field_validator("losses")
    @classmethod
    def _check_losses(cls, values: list[str]) -> list[str]:
        unknown = [v for v in values if v not in ALL_LOSS_CATEGORIES]
        if unknown:
            raise ValueError(
                f"Unknown loss categories: {unknown}. "
                f"Allowed: {ALL_LOSS_CATEGORIES}"
            )
        return values

    @field_validator("psla_band")
    @classmethod
    def _check_band(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("psla_band must be [min, max]")
        if value[0] > value[1]:
            raise ValueError("psla_band min must be <= max")
        return value

    @field_validator("overall_category")
    @classmethod
    def _check_severity(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = {c.value for c in InjuryCategory}
        if value not in allowed:
            raise ValueError(
                f"Unknown overall_category {value!r}. Allowed: {sorted(allowed)}"
            )
        return value

    def resolved_injuries(self, taxonomy: IcdTaxonomy) -> tuple[str, ...]:
        """ICD codes from explicit codes plus free-text descriptions."""
        codes: list[str] = []
        for code in self.injuries:
            if taxonomy.is_known(code):
                codes.append(code)
            else:
                resolved = resolve_injury_text(code, taxonomy)
                if resolved:
                    codes.append(resolved)
        index = _description_index(taxonomy)
        for text in self.injury_descriptions:
            resolved = resolve_injury_text(text, taxonomy, index=index)
            if resolved:
                codes.append(resolved)
        # de-duplicate, preserve order
        seen: set[str] = set()
        out: list[str] = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                out.append(code)
        return tuple(out)

    def to_corpus_case(self, taxonomy: IcdTaxonomy) -> CorpusCase:
        injuries = self.resolved_injuries(taxonomy)
        return CorpusCase(
            case_id=self.query_id,
            gender=self.gender,
            age_at_accident=self.age_at_accident,
            age_at_trial=self.age_at_trial,
            injuries=injuries,
            losses=tuple(self.losses),
            psla_nominal=None,
            year=None,
            psla_real=self.psla_target_real,
            overall_category=self.overall_category,
            operations_count=self.operations_count,
            hospitalisation_days=self.hospitalisation_days,
            n_injuries=len(injuries),
            n_losses=len(self.losses),
        )


def load_query(path: str | Path) -> FeatureQuery:
    """Load a :class:`FeatureQuery` from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        data: Any = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"Query file must contain a mapping: {path}")
    return FeatureQuery.model_validate(data)


# ---------------------------------------------------------------------------
# Free-text injury resolution
# ---------------------------------------------------------------------------
def _description_index(taxonomy: IcdTaxonomy) -> dict[str, str]:
    """Map ICD code -> description from the taxonomy's own tree."""
    # ``IcdTaxonomy`` stores descriptions privately; rebuild the mapping from
    # the same JSON so this module stays independent of its internals.
    from hklii_psla.experiments.experiment4.icd_similarity import DEFAULT_ICD_PATH

    tree = json.loads(Path(DEFAULT_ICD_PATH).read_text(encoding="utf-8"))
    index: dict[str, str] = {}

    def walk(node: dict[str, Any]) -> None:
        code = node.get("code")
        if code:
            index[code] = node.get("description", "")
        for child in node.get("children", []):
            walk(child)

    for section in tree:
        walk(section)
    return index


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOPWORDS
    }


def resolve_injury_text(
    text: str,
    taxonomy: IcdTaxonomy,
    index: dict[str, str] | None = None,
    min_score: float = 0.5,
) -> str | None:
    """Best-effort mapping of a free-text injury description to an ICD code.

    Scores every known code by token overlap with its description and returns
    the best code above ``min_score`` (or ``None``).  This is a cheap,
    dependency-free bridge for judge queries; a full LLM ICD classification
    remains the higher-quality path.
    """
    if not text or not text.strip():
        return None
    if taxonomy.is_known(text.strip()):
        return text.strip()

    query_tokens = _tokens(text)
    if not query_tokens:
        return None

    descriptions = index if index is not None else _description_index(taxonomy)
    best_code: str | None = None
    best_score = 0.0
    for code, description in descriptions.items():
        desc_tokens = _tokens(description)
        if not desc_tokens:
            continue
        overlap = len(query_tokens & desc_tokens)
        if overlap == 0:
            continue
        # Recall-weighted: how much of the query description is covered.
        score = overlap / len(query_tokens)
        if score > best_score:
            best_score = score
            best_code = code
    if best_score < min_score:
        return None
    return best_code
