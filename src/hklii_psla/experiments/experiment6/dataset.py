"""Experiment 6 — join extracted ``Case`` JSON files with the human severity labels.

Features come from the LLM-extracted JSON; the label comes from the
human-annotated CSV column (default ``injury_type``). A case is matched to a
CSV row by neutral citation, then by the filename, then by action number.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from hklii_psla.schemas import Case

_NULL_VALUES = {"", "null", "none", "na", "n/a", "nan"}
# raw judgment filenames look like HKDC_2020_203.html.txt (court_year_number)
_SOURCE_NAME_RE = re.compile(r"^(HK[A-Z]+)_(\d{4})_(\d+)")


def norm_key(value: str | None) -> str:
    """Lowercase alphanumerics only: '[2020] HKDC 203' and '2020_HKDC_203' agree."""
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def filename_keys(stem: str) -> list[str]:
    keys = [norm_key(stem.split(".")[0])]
    m = _SOURCE_NAME_RE.match(stem)
    if m:
        court, year, number = m.groups()
        keys.append(norm_key(f"{year}{court}{number}"))
    return [k for k in keys if k]


def case_keys(case: Case, stem: str) -> list[tuple[str, str]]:
    """Candidate (source, key) pairs for a case, most reliable first."""
    keys: list[tuple[str, str]] = []
    if case.metadata.neutral_citation:
        keys.append(("citation", norm_key(case.metadata.neutral_citation)))
    keys.extend(("filename", k) for k in filename_keys(stem))
    if case.metadata.action_number:
        keys.append(("action_number", norm_key(case.metadata.action_number)))
    return [(src, k) for src, k in keys if k]


@dataclass
class LabelledCase:
    case_id: str
    case: Case
    raw_label: str
    tier: str
    matched_on: str


@dataclass
class JoinReport:
    n_case_files: int = 0
    n_unreadable: int = 0
    n_csv_rows: int = 0
    n_csv_rows_without_usable_label: int = 0
    matched_on: dict[str, int] = field(default_factory=dict)
    unmatched_cases: list[str] = field(default_factory=list)
    conflicting_cases: list[str] = field(default_factory=list)
    duplicate_cases: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_case_files": self.n_case_files,
            "n_unreadable": self.n_unreadable,
            "n_csv_rows": self.n_csv_rows,
            "n_csv_rows_without_usable_label": self.n_csv_rows_without_usable_label,
            "n_matched": sum(self.matched_on.values()),
            "matched_on": self.matched_on,
            "unmatched_cases": self.unmatched_cases,
            "conflicting_cases": self.conflicting_cases,
            "duplicate_cases": self.duplicate_cases,
            "unreadable_files": self.unreadable_files,
        }


def tier_lookup(tiers: dict[str, list[str]]) -> dict[str, str]:
    """Invert ``{tier: [raw labels]}`` into ``{raw label: tier}`` (lowercased)."""
    lookup: dict[str, str] = {}
    for tier, raws in tiers.items():
        for raw in raws:
            key = raw.strip().lower()
            if key in lookup and lookup[key] != tier:
                raise ValueError(f"Label {raw!r} is assigned to two tiers: {lookup[key]!r}, {tier!r}")
            lookup[key] = tier
    return lookup


def _index_labels(
    csv_path: Path, label_column: str, lookup: dict[str, str], report: JoinReport
) -> dict[str, dict[str, set[str]]]:
    """{'citation'|'action_number': {key: {raw labels}}} for rows with a usable label."""
    index: dict[str, dict[str, set[str]]] = {"citation": {}, "action_number": {}}
    with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or label_column not in reader.fieldnames:
            raise KeyError(f"Label column {label_column!r} not in {csv_path.name}: {reader.fieldnames}")
        for row in reader:
            report.n_csv_rows += 1
            raw = (row[label_column] or "").strip().lower()
            if raw in _NULL_VALUES or raw not in lookup:
                report.n_csv_rows_without_usable_label += 1
                continue
            for source, column in (("citation", "neutral_citation"), ("action_number", "action_number")):
                key = norm_key(row.get(column))
                if key:
                    index[source].setdefault(key, set()).add(raw)
    return index


def load_labelled_cases(
    cases_path: Path,
    cases_glob: str,
    labels_path: Path,
    label_column: str,
    tiers: dict[str, list[str]],
) -> tuple[list[LabelledCase], JoinReport]:
    lookup = tier_lookup(tiers)
    report = JoinReport()
    index = _index_labels(labels_path, label_column, lookup, report)

    files = sorted(cases_path.glob(cases_glob))
    report.n_case_files = len(files)
    labelled: list[LabelledCase] = []
    used_rows: set[tuple[str, str]] = set()  # one labelled row per CSV judgment

    for path in files:
        try:
            case = Case.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # unreadable / not a Case: reported, never fatal
            report.n_unreadable += 1
            reason = str(exc).strip().splitlines()
            report.unreadable_files.append(f"{path.name}: {' '.join(reason[1:3]) or reason[0] if reason else exc!r}"[:240])
            continue

        hit: tuple[str, str, set[str]] | None = None
        for source, key in case_keys(case, path.stem):
            # filename keys are looked up against the citation index
            found = index["action_number" if source == "action_number" else "citation"].get(key)
            if found:
                hit = (source, key, found)
                break

        if hit is None:
            report.unmatched_cases.append(path.name)
            continue
        source, key, raws = hit
        row_id = ("action_number" if source == "action_number" else "citation", key)
        if len(raws) > 1:
            report.conflicting_cases.append(path.name)
            continue
        if row_id in used_rows:
            report.duplicate_cases.append(path.name)
            continue
        used_rows.add(row_id)
        raw = next(iter(raws))
        labelled.append(LabelledCase(path.stem, case, raw, lookup[raw], source))
        report.matched_on[source] = report.matched_on.get(source, 0) + 1

    return labelled, report
