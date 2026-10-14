"""Experiment 7 — plain-language names for the feature columns.

Turns ``icd4_NB32`` into "ICD-11 NB32: Injury of other or unspecified intrathoracic organs"
so the report can be read without the feature glossary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

_STATIC = {
    "n_injuries": "Number of injuries",
    "n_injuries_without_icd": "Injuries the ICD-11 step could not code",
    "n_distinct_icd_blocks": "Number of different ICD-11 body blocks injured",
    "n_bilateral_injuries": "Number of bilateral injuries",
    "has_pre_existing_injuries": "Pre-existing injuries noted",
    "n_manifestations": "Number of symptoms / functional deficits",
    "hospitalisation_days": "Hospital stay, days",
    "expected_hospitalisation_days": "Expected future hospital stay, days",
    "operations_count": "Number of operations received",
    "future_operations_count": "Number of future operations expected",
    "sick_leave_days_actual": "Sick leave taken, days",
    "sick_leave_days_expected": "Expected sick leave, days",
    "age_at_accident": "Age at accident",
    "male": "Plaintiff is male",
    "salary_before": "Monthly salary before accident",
    "salary_after": "Monthly salary after accident",
    "salary_drop_ratio": "Share of salary lost",
    "occupation_changed": "Occupation changed after accident",
    "n_losses": "Number of losses claimed",
    "n_loss_categories": "Number of different loss categories",
    "n_treatments_received": "Number of treatments received",
    "n_treatments_future": "Number of future treatments",
    "n_surgical_received": "Number of surgical treatments received",
    "n_surgical_future": "Number of surgical treatments expected",
}

_VMIS_TIERS = {"minor": "Minor", "intermediate": "Intermediate", "major": "Major", "complex": "Complex"}


def _icd_index(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """code -> description, and 2-letter block -> top-level section name."""
    tree = json.loads(path.read_text(encoding="utf-8"))
    desc: dict[str, str] = {}
    block: dict[str, str] = {}

    def walk(node: dict, section: str) -> None:
        code = node.get("code")
        if code:
            desc.setdefault(code, node.get("description", ""))
            block.setdefault(code[:2], section)
        for child in node.get("children", []):
            walk(child, section)

    for top in tree:
        walk(top, top.get("section", ""))
    return desc, block


def make_explainer(icd_path: Path) -> Callable[[str], str]:
    try:
        icd_desc, icd_block = _icd_index(icd_path)
    except (OSError, ValueError):
        icd_desc, icd_block = {}, {}

    def explain(feature: str) -> str:
        if feature in _STATIC:
            return _STATIC[feature]
        if feature.endswith("_stated"):
            base = feature.removesuffix("_stated")
            return f"{_STATIC.get(base, base)} is stated in the judgment"
        if m := re.fullmatch(r"icd2_([A-Z0-9]{2})", feature):
            code = m.group(1)
            return f"ICD-11 block {code}: {icd_block.get(code, 'unknown section')}"
        if m := re.fullmatch(r"icd4_([A-Z0-9]{4})", feature):
            code = m.group(1)
            hit = icd_desc.get(code) or next((d for c, d in icd_desc.items() if c.startswith(code)), "")
            return f"ICD-11 {code}: {hit or 'unknown'}"
        if m := re.fullmatch(r"vmis_max_tier_(received|future)", feature):
            return f"Highest VMIS surgery tier {m.group(1)} (0 none, 1 Minor ... 4 Complex)"
        if m := re.fullmatch(r"vmis_unmapped_(received|future)", feature):
            return f"Surgeries {m.group(1)} that are not in the VMIS schedule"
        if m := re.fullmatch(r"vmis_(\w+?)_(received|future)", feature):
            return f"Number of {_VMIS_TIERS.get(m.group(1), m.group(1))} VMIS surgeries {m.group(2)}"
        for prefix, label in (
            ("injkw_", "Injury mentions"), ("treatkw_", "Treatment includes"),
            ("manif_", "Manifestations of type"), ("injtype_", "Injuries of type"),
            ("losscat_", "Loss claimed:"),
        ):
            if feature.startswith(prefix):
                return f"{label} {feature.removeprefix(prefix).replace('_', ' ')}"
        return feature.replace("_", " ")

    return explain
