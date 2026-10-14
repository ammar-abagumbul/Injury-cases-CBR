"""Experiment 6 — turn an extracted ``Case`` into a flat numeric feature row.

Features are organised in named groups so experiments can ablate them:

    injuries      ICD-11 block counts, injury types, keyword flags, manifestations
    treatment     treatment counts, VMIS surgical tiers, treatment keyword flags
    hospital_sick hospital stay, operations and sick leave (value + "stated" flag)
    demographics  age, gender, pre-existing injury
    economic      salary and occupation change
    losses        loss-of-amenity categories claimed

Deliberately NOT used (they would leak the answer): ``psla`` awards, comparable
cases, and the LLM's own ``injuries.overall_category``.

Two kinds of column exist. *Sparse* columns (counts / flags that are naturally
0 when absent, identified by ``SPARSE_PREFIXES``) are filled with 0. *Dense*
columns (days, age, salary ...) stay NaN when the judgment does not state them
and are paired with a ``*_stated`` flag, so a model can tell "0" from "unknown".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from hklii_psla.schemas import Case, TreatmentItem

FEATURE_GROUPS = ("injuries", "treatment", "hospital_sick", "demographics", "economic", "losses")

SPARSE_PREFIXES = ("icd2_", "icd4_", "injtype_", "injkw_", "manif_", "vmis_", "treatkw_", "losscat_")

# heavy-tailed, non-negative columns that benefit from log1p in linear models
LOG_COLUMNS = (
    "hospitalisation_days", "expected_hospitalisation_days",
    "sick_leave_days_actual", "sick_leave_days_expected",
    "operations_count", "future_operations_count",
    "salary_before", "salary_after",
)

_TIER_ORDER = {"Minor": 1, "Intermediate": 2, "Major": 3, "Complex": 4}

_INJURY_KEYWORDS = {
    "fracture": r"fractur",
    "dislocation": r"dislocat",
    "amputation": r"amputat|loss of (a |the )?(finger|toe|limb|arm|leg)",
    "head_brain": r"brain|skull|intracranial|concussion|head injur",
    "spinal": r"spinal|vertebra|spine|disc|cord",
    "internal_organ": r"spleen|liver|kidney|lung|bowel|pneumothorax|internal (injur|bleed)",
}
_TREATMENT_KEYWORDS = {
    "physio": r"physio|rehabilitat",
    "psych": r"psych|counsel|cognitive",
    "cast_splint": r"\bcast\b|splint|brace|plaster|sling",
    "icu": r"intensive care|\bicu\b|ventilat|intubat",
    "medication": r"medicat|painkiller|analges|drug",
    "injection": r"inject",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _value(x) -> str:
    return getattr(x, "value", x)


def _number(x) -> float:
    return float("nan") if x is None else float(x)


def _stated(x) -> float:
    return 0.0 if x is None else 1.0


def _any_match(pattern: str, texts: Iterable[str]) -> float:
    return float(any(re.search(pattern, t, re.IGNORECASE) for t in texts))


def _injury_features(case: Case) -> dict[str, float]:
    injuries = case.injuries.injuries
    f: dict[str, float] = {
        "n_injuries": float(len(injuries)),
        "n_injuries_without_icd": float(sum(1 for i in injuries if not i.icd_code)),
        "n_distinct_icd_blocks": float(len({i.icd_code[:2] for i in injuries if i.icd_code})),
        "n_bilateral_injuries": float(sum(1 for i in injuries if i.laterality == "bilateral")),
        "has_pre_existing_injuries": float(bool(case.metadata.has_pre_existing_injuries)),
    }
    for i in injuries:
        if i.icd_code:
            for n in (2, 4):
                if len(i.icd_code) >= n:
                    key = f"icd{n}_{i.icd_code[:n].upper()}"
                    f[key] = f.get(key, 0.0) + 1.0
        key = f"injtype_{_slug(_value(i.injury_type))}"
        f[key] = f.get(key, 0.0) + 1.0

    texts = [f"{i.description} {i.icd_description or ''}" for i in injuries]
    for name, pattern in _INJURY_KEYWORDS.items():
        f[f"injkw_{name}"] = _any_match(pattern, texts)

    f["n_manifestations"] = float(len(case.injuries.manifestations))
    for m in case.injuries.manifestations:
        key = f"manif_{_slug(_value(m.manifestation_type))}"
        f[key] = f.get(key, 0.0) + 1.0
    return f


def _tier_features(items: list[TreatmentItem], suffix: str) -> dict[str, float]:
    surgical = [t for t in items if t.is_surgical]
    f = {
        f"n_treatments_{suffix}": float(len(items)),
        f"n_surgical_{suffix}": float(len(surgical)),
        f"vmis_unmapped_{suffix}": float(sum(1 for t in surgical if t.vmis_category is None)),
        f"vmis_max_tier_{suffix}": float(
            max((_TIER_ORDER[_value(t.vmis_category)] for t in surgical if t.vmis_category), default=0)
        ),
    }
    for t in surgical:
        if t.vmis_category is not None:
            key = f"vmis_{_slug(_value(t.vmis_category))}_{suffix}"
            f[key] = f.get(key, 0.0) + 1.0
    return f


def _treatment_features(case: Case) -> dict[str, float]:
    tr = case.treatment
    f: dict[str, float] = {}
    f.update(_tier_features(tr.treatments_received, "received"))
    f.update(_tier_features(tr.treatments_future, "future"))
    texts = [t.treatment for t in tr.treatments_received + tr.treatments_future]
    for name, pattern in _TREATMENT_KEYWORDS.items():
        f[f"treatkw_{name}"] = _any_match(pattern, texts)
    return f


def _hospital_sick_features(case: Case) -> dict[str, float]:
    tr = case.treatment
    f: dict[str, float] = {}
    for name in (
        "hospitalisation_days", "expected_hospitalisation_days",
        "operations_count", "future_operations_count",
        "sick_leave_days_actual", "sick_leave_days_expected",
    ):
        value = getattr(tr, name)
        f[name] = _number(value)
        f[f"{name}_stated"] = _stated(value)
    return f


def _demographic_features(case: Case) -> dict[str, float]:
    p = case.plaintiff
    gender = _value(p.gender) if p.gender else None
    return {
        "age_at_accident": _number(p.age_at_accident),
        "age_at_accident_stated": _stated(p.age_at_accident),
        "male": float("nan") if gender is None else float(str(gender).lower() == "male"),
    }


def _economic_features(case: Case) -> dict[str, float]:
    p = case.plaintiff
    before, after = p.salary_before, p.salary_after
    drop = float("nan")
    if before and after is not None and before > 0:
        drop = 1.0 - after / before
    changed = float("nan")
    if p.occupation_before and p.occupation_after:
        changed = float(p.occupation_before.strip().lower() != p.occupation_after.strip().lower())
    return {
        "salary_before": _number(before),
        "salary_after": _number(after),
        "salary_drop_ratio": drop,
        "occupation_changed": changed,
    }


def _loss_features(case: Case) -> dict[str, float]:
    losses = case.losses.losses
    f: dict[str, float] = {"n_losses": float(len(losses))}
    categories = {_value(loss.category) for loss in losses}
    f["n_loss_categories"] = float(len(categories))
    for loss in losses:
        key = f"losscat_{_slug(_value(loss.category))}"
        f[key] = f.get(key, 0.0) + 1.0
    return f


_EXTRACTORS = {
    "injuries": _injury_features,
    "treatment": _treatment_features,
    "hospital_sick": _hospital_sick_features,
    "demographics": _demographic_features,
    "economic": _economic_features,
    "losses": _loss_features,
}


def extract_features(case: Case) -> dict[str, dict[str, float]]:
    """Features of one case, grouped by feature group."""
    return {group: fn(case) for group, fn in _EXTRACTORS.items()}


def build_feature_table(
    cases: list[Case], min_support: int = 1
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Feature matrix (one row per case) and ``{group: [columns]}``.

    Sparse columns are filled with 0, dense columns keep NaN. Sparse columns
    that are non-zero in fewer than ``min_support`` cases are dropped (this uses
    no labels, so it cannot leak).
    """
    rows = [extract_features(c) for c in cases]
    column_group: dict[str, str] = {}
    flat: list[dict[str, float]] = []
    for row in rows:
        merged: dict[str, float] = {}
        for group, feats in row.items():
            for col, val in feats.items():
                column_group.setdefault(col, group)
                merged[col] = val
        flat.append(merged)

    df = pd.DataFrame(flat, index=range(len(cases)))
    sparse = [c for c in df.columns if c.startswith(SPARSE_PREFIXES)]
    df[sparse] = df[sparse].fillna(0.0)

    drop = [c for c in sparse if (df[c] != 0).sum() < min_support]
    df = df.drop(columns=drop)

    groups: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}
    for col in df.columns:
        groups[column_group[col]].append(col)
    df = df.astype(float)
    return df, groups


def llm_overall_tier(case: Case) -> str | None:
    """The LLM's own ``overall_category`` (raw lowercase label), used only as a baseline."""
    cat = case.injuries.overall_category
    if cat is None:
        return None
    return {
        "Non-serious injury": "nonserious",
        "Serious injury": "serious",
        "Substantial injury": "substantial",
        "Gross disability": "gross",
        "Disaster": "disaster",
    }.get(_value(cat))


__all__ = [
    "FEATURE_GROUPS", "LOG_COLUMNS", "SPARSE_PREFIXES",
    "build_feature_table", "extract_features", "llm_overall_tier",
]
