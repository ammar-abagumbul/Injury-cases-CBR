"""Deterministic corpus QC audit for the Experiment 5 extractions.

This module performs **no LLM calls**. It walks the extracted ``Case`` JSONs
produced by ``experiment5/exp_5.py`` and emits a triage table describing which
cases need attention and why.

Outputs (under ``--out``, default ``output/experiments/corpus_extraction/qc``):

* ``qc.csv``      — one row per case, with a ``flags`` column and counts.
* ``qc_report.json`` — aggregate flag counts plus worst-offending examples.

Run with::

    uv run python -m hklii_psla.experiments.experiment6.audit
    uv run psla audit-corpus --limit 20

The named flags are intentionally fine-grained so the partial-edit agent can
consume them directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]

DEFAULT_CASES_DIR = REPO_ROOT / "output/experiments/corpus_extraction/cases"
DEFAULT_PROGRESS = REPO_ROOT / "output/experiments/corpus_extraction/progress.csv"
DEFAULT_SOURCE_DIR = REPO_ROOT / "old_pi_cases_web_text"
DEFAULT_ICD_TREE = REPO_ROOT / "data/ICD-11.json"
DEFAULT_OUT_DIR = REPO_ROOT / "output/experiments/corpus_extraction/qc"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Config:
    coarse: dict[str, Any]
    red_flags: dict[str, Any]
    psych: dict[str, Any]
    qc: dict[str, Any]

    @classmethod
    def load(cls, base: Path = HERE) -> Config:
        return cls(
            coarse=_load_json(base / "coarse_nodes.json"),
            red_flags=_load_json(base / "red_flag_fields.json"),
            psych=_load_json(base / "psychiatric_keywords.json"),
            qc=_load_json(base / "qc_config.json"),
        )


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def _make_matcher(
    keywords: list[str], exclusions: list[str] | None = None
) -> Callable[[str | None], str | None]:
    """Return a case-insensitive substring matcher honouring exclusions.

    Exclusions are removed from the haystack before matching, so
    ``depressed skull fracture`` does not trigger the psychiatric ``depress*``
    keywords.
    """
    exclusions = exclusions or []
    lowered = [k.lower() for k in keywords if k]
    excl = [re.compile(re.escape(x.lower()), re.IGNORECASE) for x in exclusions]

    def match(text: str | None) -> str | None:
        if not text:
            return None
        hay = text.lower()
        for e in excl:
            hay = e.sub(" ", hay)
        for k in lowered:
            if k in hay:
                return k
        return None

    return match


# ---------------------------------------------------------------------------
# Citation recovery from the source header
# ---------------------------------------------------------------------------

NC_RE = re.compile(r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)")
ACTION_RE = re.compile(r"\b([A-Z]{2,6})\s*0*(\d+)\s*/\s*(\d{4})\b")


def source_header_block(text: str) -> str:
    """Return the reflowed first markdown ``##`` heading of a source document.

    HKLII headers are hard-wrapped across lines, e.g.::

        ##  CHEUNG YUK CHUN v. ... [1984]
        HKCFI 403; HCA 12597/1982 (8 November 1984)

    so the block is joined and whitespace-collapsed before parsing.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("##"):
            buf: list[str] = []
            for j in range(i, len(lines)):
                if j > i and not lines[j].strip():
                    break
                buf.append(lines[j])
            return re.sub(r"\s+", " ", " ".join(buf)).strip()
    return ""


def parse_citation_from_source(text: str) -> tuple[str | None, str | None]:
    """Recover ``(neutral_citation, action_number)`` from a source document."""
    header = source_header_block(text)
    nc_match = NC_RE.search(header)
    action_match = ACTION_RE.search(header)
    neutral = (
        f"[{nc_match.group(1)}] {nc_match.group(2)} {nc_match.group(3)}"
        if nc_match
        else None
    )
    action = (
        f"{action_match.group(1)} {int(action_match.group(2))}/{action_match.group(3)}"
        if action_match
        else None
    )
    return neutral, action


# Neutral citations only ever use a small set of court codes; law-report series
# such as ``HKC`` / ``HKLRD`` also match ``NC_RE`` but must never be mistaken
# for a neutral citation.  Recovery is therefore restricted to an allowlist.
DEFAULT_COURTS: tuple[str, ...] = ("HKCFI", "HKDC", "HKCA", "UKPC")


def make_citation_recovery_re(courts: list[str] | tuple[str, ...]) -> re.Pattern[str]:
    """Compile a recovery regex matching ``[YYYY] COURT NUM`` for known courts."""
    alternation = "|".join(re.escape(c) for c in courts)
    return re.compile(rf"\[(\d{{4}})\]\s+({alternation})\s+(\d+)")


def recover_citation_from_extracted(
    neutral_citation: str | None, pattern: re.Pattern[str]
) -> str | None:
    """Recover ``[YYYY] COURT NUM`` from an already-extracted citation field.

    The extraction frequently captures extra text after — or a party-name
    prefix before — an otherwise correct citation, e.g.
    ``"[1989] HKCFI 191; HCAJ 209/1984 (10 November 1989)"`` or
    ``"SIT KA YEE v. LAI WAI HO [2001] HKDC 52"``.  ``search`` (not ``match``)
    is used so a leading prefix does not defeat recovery.
    """
    if not neutral_citation:
        return None
    match = pattern.search(neutral_citation)
    if not match:
        return None
    return f"[{match.group(1)}] {match.group(2)} {match.group(3)}"


@dataclass
class CitationResolution:
    """Outcome of resolving a case's ``neutral_citation``.

    ``outcome`` is one of:

    * ``clean``         — the extracted field is already exactly a citation.
    * ``extra_text``    — a citation was recovered from the extracted field,
      which carried extra text (prefix/suffix) that must be stripped.
    * ``from_source``   — no citation could be recovered from the extracted
      field; the source header supplied it instead.
    * ``conflict``      — the extraction and the source disagree; left for
      review rather than silently overwritten.
    * ``unrecoverable`` — neither the extraction nor the source yielded one.
    """

    recovered: str | None = None
    source: str | None = None
    source_action: str | None = None
    resolved: str | None = None
    outcome: str = "unrecoverable"

    @property
    def needs_source(self) -> bool:
        return self.outcome in ("from_source", "conflict")


def resolve_citation(
    neutral_citation: str | None,
    source_text: str,
    pattern: re.Pattern[str],
) -> CitationResolution:
    """Decide the best neutral citation for a case (extraction-first).

    The extracted field is always preferred when it yields a citation; the
    source header is parsed regardless (it is cheap and deterministic) so that
    an extraction/source disagreement can be surfaced as a conflict.
    """
    current = (neutral_citation or "").strip()
    recovered = recover_citation_from_extracted(current, pattern)
    source, source_action = parse_citation_from_source(source_text)
    result = CitationResolution(
        recovered=recovered,
        source=source,
        source_action=source_action,
    )

    if recovered and source and recovered != source:
        result.outcome = "conflict"
        result.resolved = None  # leave for review
    elif recovered:
        result.outcome = "clean" if recovered == current else "extra_text"
        result.resolved = recovered
    elif source:
        result.outcome = "from_source"
        result.resolved = source
    else:
        result.outcome = "unrecoverable"
    return result


# ---------------------------------------------------------------------------
# Per-case audit
# ---------------------------------------------------------------------------

SCALAR_NULL = (None, "", [], {})


@dataclass
class CaseAudit:
    source_file: str
    output_json: str
    neutral_citation: str = ""
    canonical_citation: str | None = None
    canonical_action: str | None = None
    recovered_citation: str | None = None
    citation_outcome: str = ""
    flags: list[str] = field(default_factory=list)
    red_flag_fields: list[str] = field(default_factory=list)
    null_ratio: float = 0.0
    n_injuries: int = 0
    n_null_icd: int = 0
    n_psych: int = 0
    n_coarse_icd: int = 0
    n_stopped_early: int = 0
    n_not_in_tree: int = 0
    n_multi_body_part: int = 0
    n_unlinked_manifestations: int = 0
    n_unlinked_losses: int = 0
    n_null_buckets: dict[str, int] = field(default_factory=dict)
    psych_injury_ids: list[str] = field(default_factory=list)
    priority: int = 0
    error: str | None = None

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def to_row(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "output_json": self.output_json,
            "neutral_citation": self.neutral_citation,
            "canonical_citation": self.canonical_citation or "",
            "canonical_action": self.canonical_action or "",
            "recovered_citation": self.recovered_citation or "",
            "citation_outcome": self.citation_outcome,
            "priority": self.priority,
            "flags": ";".join(self.flags),
            "red_flag_fields": ";".join(self.red_flag_fields),
            "null_ratio": f"{self.null_ratio:.3f}",
            "n_injuries": self.n_injuries,
            "n_null_icd": self.n_null_icd,
            "n_psych": self.n_psych,
            "n_coarse_icd": self.n_coarse_icd,
            "n_stopped_early": self.n_stopped_early,
            "n_not_in_tree": self.n_not_in_tree,
            "n_multi_body_part": self.n_multi_body_part,
            "n_unlinked_manifestations": self.n_unlinked_manifestations,
            "n_unlinked_losses": self.n_unlinked_losses,
            "n_null_psych": self.n_null_buckets.get("psych", 0),
            "n_null_preexisting": self.n_null_buckets.get("preexisting", 0),
            "n_null_burn": self.n_null_buckets.get("burn", 0),
            "n_null_symptom": self.n_null_buckets.get("symptom", 0),
            "n_null_other": self.n_null_buckets.get("other", 0),
            "psych_injury_ids": ",".join(self.psych_injury_ids),
            "error": self.error or "",
        }


def _iter_get(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _is_null(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


class CorpusAuditor:
    def __init__(
        self,
        config: Config,
        cases_dir: Path,
        progress_path: Path,
        source_dir: Path,
        icd_tree_path: Path,
    ) -> None:
        self.config = config
        self.cases_dir = cases_dir
        self.progress_path = progress_path
        self.source_dir = source_dir

        self.by_code, self.parent_of = self._index_icd_tree(icd_tree_path)
        self.coarse_nodes: dict[str, Any] = config.coarse["nodes"]

        # matchers
        self.psych_match = _make_matcher(
            config.psych["keywords"], config.psych.get("exclusions", [])
        )
        buckets = config.qc["null_icd_buckets"]
        self.bucket_matchers = {
            "preexisting": _make_matcher(buckets["preexisting"]["keywords"]),
            "burn": _make_matcher(buckets["burn"]["keywords"]),
            "symptom": _make_matcher(buckets["symptom"]["keywords"]),
        }
        self.treatment_re = [
            re.compile(p, re.IGNORECASE)
            for p in config.qc["treatment_contamination"]["patterns"]
        ]
        self.noise_re = [
            re.compile(p, re.IGNORECASE) for p in config.qc["source_noise"]["patterns"]
        ]
        self.multi_bp_re = [
            re.compile(re.escape(p), re.IGNORECASE)
            for p in config.qc["multi_body_part"]["patterns"]
        ]
        self.nc_pattern = re.compile(
            config.qc["citation"]["neutral_citation_pattern"]
        )
        courts = config.qc["citation"].get("court_allowlist") or list(DEFAULT_COURTS)
        self.citation_recovery_re = make_citation_recovery_re(courts)

    # -- ICD index -------------------------------------------------------
    @staticmethod
    def _index_icd_tree(path: Path) -> tuple[dict[str, dict], dict[str, str | None]]:
        tree = _load_json(path)
        by_code: dict[str, dict] = {}
        parent_of: dict[str, str | None] = {}

        def walk(node: dict[str, Any], parent: str | None) -> None:
            code = node.get("code")
            if code:
                by_code[code] = node
                parent_of[code] = parent
            for child in node.get("children", []):
                walk(child, code or parent)

        for section in tree:
            walk(section, None)
        return by_code, parent_of

    # -- main loop -------------------------------------------------------
    def audit(self, only: set[str] | None = None, limit: int | None = None) -> list[CaseAudit]:
        rows = list(csv.DictReader(self.progress_path.open(encoding="utf-8")))
        results: list[CaseAudit] = []
        for row in rows:
            output_json = (row.get("output_json_file") or "").strip()
            if not output_json:
                continue
            case_path = self.cases_dir / output_json
            if not case_path.exists():
                continue
            if only and output_json not in only and row["source_file"] not in only:
                continue
            results.append(self.audit_case(row["source_file"], output_json, case_path))
            if limit is not None and len(results) >= limit:
                break
        return results

    def audit_case(
        self, source_file: str, output_json: str, case_path: Path
    ) -> CaseAudit:
        audit = CaseAudit(source_file=source_file, output_json=output_json)
        try:
            case = _load_json(case_path)
        except Exception as exc:  # noqa: BLE001
            audit.error = f"load error: {exc}"
            audit.add_flag("JSON_UNREADABLE")
            audit.priority = 100
            return audit

        self._audit_metadata(audit, case, source_file)
        self._audit_red_flags(audit, case)
        self._audit_injuries(audit, case)
        self._audit_relations(audit, case)
        self._audit_treatment(audit, case)
        self._audit_source_noise(audit, source_file)
        self._audit_psych(audit, case)
        self._audit_counts(audit, case)
        audit.priority = self._priority(audit)
        return audit

    # -- metadata / citation --------------------------------------------
    def _audit_metadata(
        self, audit: CaseAudit, case: dict[str, Any], source_file: str
    ) -> None:
        meta = case.get("metadata", {})
        nc = (meta.get("neutral_citation") or "").strip()
        audit.neutral_citation = nc

        text = ""
        src_path = self.source_dir / source_file
        if src_path.exists():
            try:
                text = src_path.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                text = ""

        resolution = resolve_citation(nc, text, self.citation_recovery_re)
        audit.recovered_citation = resolution.recovered
        audit.canonical_citation = resolution.source
        audit.canonical_action = resolution.source_action
        audit.citation_outcome = resolution.outcome

        if not nc:
            audit.add_flag("CITATION_MISSING")
        elif not self.nc_pattern.match(nc):
            audit.add_flag("CITATION_MALFORMED")
            if resolution.recovered:
                audit.add_flag("CITATION_EXTRA_TEXT")

        if resolution.source and nc and resolution.source != nc:
            audit.add_flag("CITATION_MISMATCH")
        if resolution.outcome == "conflict":
            audit.add_flag("CITATION_CONFLICT")
        elif resolution.outcome == "from_source":
            audit.add_flag("CITATION_FROM_SOURCE")

        action = (meta.get("action_number") or "").strip()
        if not action:
            audit.add_flag("ACTION_MISSING")

    # -- red-flag nulls --------------------------------------------------
    def _audit_red_flags(self, audit: CaseAudit, case: dict[str, Any]) -> None:
        for spec in self.config.red_flags["fields"]:
            path = spec["path"]
            if _is_null(_iter_get(case, path)):
                audit.red_flag_fields.append(path)
                severity = spec.get("severity", "low")
                if severity == "high":
                    audit.add_flag("RED_FLAG_NULL")
                elif severity == "medium":
                    audit.add_flag("RED_FLAG_NULL_MEDIUM")

        scalar_fields = self.config.qc["null_ratio"]["scalar_fields"]
        nulls = sum(1 for f in scalar_fields if _is_null(_iter_get(case, f)))
        audit.null_ratio = nulls / len(scalar_fields) if scalar_fields else 0.0
        if audit.null_ratio >= self.config.qc["null_ratio"]["threshold"]:
            audit.add_flag("NULL_RATIO_HIGH")

    # -- injuries --------------------------------------------------------
    def _audit_injuries(self, audit: CaseAudit, case: dict[str, Any]) -> None:
        injury_summary = case.get("injuries", {})
        injuries = injury_summary.get("injuries", []) or []
        audit.n_injuries = len(injuries)

        for injury in injuries:
            code = injury.get("icd_code")
            desc = injury.get("description") or ""
            src = " ".join(injury.get("source") or [])

            if code is None or code == "":
                audit.n_null_icd += 1
                bucket = self._null_bucket(desc, src)
                audit.n_null_buckets[bucket] = audit.n_null_buckets.get(bucket, 0) + 1
                audit.add_flag(
                    {
                        "psych": "NULL_ICD_PSYCH",
                        "preexisting": "NULL_ICD_PREEXISTING",
                        "burn": "NULL_ICD_BURN",
                        "symptom": "NULL_ICD_SYMPTOM",
                        "other": "NULL_ICD_OTHER",
                    }[bucket]
                )
            else:
                node = self.by_code.get(code)
                if node is None:
                    audit.n_not_in_tree += 1
                    audit.add_flag("ICD_CODE_NOT_IN_TREE")
                else:
                    if node.get("children"):
                        audit.n_stopped_early += 1
                        audit.add_flag("ICD_STOPPED_EARLY")
                    if code in self.coarse_nodes:
                        audit.n_coarse_icd += 1
                        audit.add_flag("COARSE_ICD_NODE")

            body_part = injury.get("body_part") or ""
            if body_part and any(p.search(body_part) for p in self.multi_bp_re):
                audit.n_multi_body_part += 1
                audit.add_flag("INJURY_MULTI_BODY_PART")

        # pre-existing metadata consistency
        has_pre = bool(case.get("metadata", {}).get("has_pre_existing_injuries"))
        if audit.n_null_buckets.get("preexisting", 0) and not has_pre:
            audit.add_flag("PREEXISTING_METADATA_MISMATCH")

    def _null_bucket(self, desc: str, src: str) -> str:
        text = f"{desc} {src}"
        if self.psych_match(text):
            return "psych"
        for bucket in ("preexisting", "burn", "symptom"):
            if self.bucket_matchers[bucket](text):
                return bucket
        return "other"

    # -- relations -------------------------------------------------------
    def _audit_relations(self, audit: CaseAudit, case: dict[str, Any]) -> None:
        injuries = case.get("injuries", {})
        for manifest in injuries.get("manifestations", []) or []:
            if not manifest.get("caused_by_injury_id"):
                audit.n_unlinked_manifestations += 1
        if audit.n_unlinked_manifestations:
            audit.add_flag("UNLINKED_MANIFESTATION")

        for loss in case.get("losses", {}).get("losses", []) or []:
            if not loss.get("caused_by_injury_id"):
                audit.n_unlinked_losses += 1
        if audit.n_unlinked_losses:
            audit.add_flag("UNLINKED_LOSS")

    # -- treatment -------------------------------------------------------
    def _audit_treatment(self, audit: CaseAudit, case: dict[str, Any]) -> None:
        treatment = case.get("treatment", {})
        items: list[str] = []
        items += treatment.get("treatments_received", []) or []
        items += treatment.get("treatments_future", []) or []
        for item in items:
            if any(r.search(item) for r in self.treatment_re):
                audit.add_flag("TREATMENT_CONTAMINATED")
                break

    # -- source noise ----------------------------------------------------
    def _audit_source_noise(self, audit: CaseAudit, source_file: str) -> None:
        src_path = self.source_dir / source_file
        if not src_path.exists():
            audit.add_flag("SOURCE_MISSING")
            return
        text = src_path.read_text(encoding="utf-8", errors="replace")
        if any(r.search(text) for r in self.noise_re):
            audit.add_flag("SOURCE_NOISE")
            return
        alpha = sum(ch.isalpha() for ch in text)
        ratio = alpha / len(text) if text else 0.0
        if ratio < self.config.qc["source_noise"]["min_alpha_ratio"]:
            audit.add_flag("SOURCE_NOISE")

    # -- psychiatric -----------------------------------------------------
    def _audit_psych(self, audit: CaseAudit, case: dict[str, Any]) -> None:
        injuries = case.get("injuries", {})
        hit = False
        for injury in injuries.get("injuries", []) or []:
            text = (injury.get("description") or "") + " " + " ".join(
                injury.get("source") or []
            )
            if self.psych_match(text):
                hit = True
                iid = injury.get("injury_id")
                if iid:
                    audit.psych_injury_ids.append(iid)
        for manifest in injuries.get("manifestations", []) or []:
            if self.psych_match(manifest.get("description")):
                hit = True
        if not hit:
            return
        audit.n_psych = 1
        audit.add_flag("PSYCH_DETECTED")
        target = self.config.psych["target_loss_category"]
        loss_cats = {
            (loss.get("category") or "")
            for loss in case.get("losses", {}).get("losses", []) or []
        }
        if target not in loss_cats:
            audit.add_flag("PSYCH_NO_PERSONALITY_CHANGE_LOSS")

    # -- simple counts ---------------------------------------------------
    def _audit_counts(self, audit: CaseAudit, case: dict[str, Any]) -> None:
        if audit.n_injuries == 0:
            audit.add_flag("ZERO_INJURIES")
        if not (case.get("losses", {}).get("losses") or []):
            audit.add_flag("ZERO_LOSSES")
        if _is_null(case.get("psla", {}).get("amount")):
            audit.add_flag("NO_PSLA_AMOUNT")
        if not (case.get("psla", {}).get("comparable_cases") or []):
            audit.add_flag("NO_COMPARABLE")
        if int(case.get("metadata", {}).get("plaintiff_count") or 1) > 1:
            audit.add_flag("MULTI_PLAINTIFF")

    # -- priority --------------------------------------------------------
    @staticmethod
    def _priority(audit: CaseAudit) -> int:
        weights = {
            "JSON_UNREADABLE": 100,
            "CITATION_MISSING": 20,
            "CITATION_MALFORMED": 15,
            "CITATION_MISMATCH": 10,
            "CITATION_CONFLICT": 16,
            "CITATION_FROM_SOURCE": 6,
            "ACTION_MISSING": 8,
            "ICD_CODE_NOT_IN_TREE": 25,
            "COARSE_ICD_NODE": 6,
            "ICD_STOPPED_EARLY": 6,
            "NULL_ICD_PSYCH": 5,
            "NULL_ICD_PREEXISTING": 4,
            "NULL_ICD_BURN": 5,
            "NULL_ICD_SYMPTOM": 6,
            "PSYCH_NO_PERSONALITY_CHANGE_LOSS": 7,
            "RED_FLAG_NULL": 10,
            "RED_FLAG_NULL_MEDIUM": 5,
            "NULL_RATIO_HIGH": 8,
            "ZERO_INJURIES": 12,
            "ZERO_LOSSES": 9,
            "TREATMENT_CONTAMINATED": 7,
            "SOURCE_NOISE": 12,
            "UNLINKED_MANIFESTATION": 2,
            "UNLINKED_LOSS": 2,
            "INJURY_MULTI_BODY_PART": 3,
            "PREEXISTING_METADATA_MISMATCH": 3,
        }
        score = 0
        for flag in audit.flags:
            score += weights.get(flag, 1)
        score += audit.n_coarse_icd + audit.n_stopped_early + audit.n_not_in_tree
        return score


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_outputs(results: list[CaseAudit], out_dir: Path, config: Config) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results = sorted(results, key=lambda r: (-r.priority, r.output_json))
    columns = list(CaseAudit("", "").to_row().keys())
    with (out_dir / "qc.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())

    flag_counts: Counter[str] = Counter()
    for result in results:
        flag_counts.update(result.flags)
    bucket_counts: Counter[str] = Counter()
    for result in results:
        bucket_counts.update(result.n_null_buckets)

    report = {
        "summary": {
            "cases_audited": len(results),
            "flagged_cases": sum(1 for r in results if r.flags),
            "clean_cases": sum(1 for r in results if not r.flags),
            "injuries": sum(r.n_injuries for r in results),
            "null_icd": sum(r.n_null_icd for r in results),
            "coarse_icd": sum(r.n_coarse_icd for r in results),
            "stopped_early": sum(r.n_stopped_early for r in results),
            "not_in_tree": sum(r.n_not_in_tree for r in results),
        },
        "flag_counts": dict(flag_counts.most_common()),
        "null_icd_bucket_counts": dict(bucket_counts.most_common()),
        "config": {
            "coarse_nodes": len(config.coarse["nodes"]),
            "red_flag_fields": len(config.red_flags["fields"]),
            "psych_keywords": len(config.psych["keywords"]),
            "null_ratio_threshold": config.qc["null_ratio"]["threshold"],
        },
        "worst_cases": [
            {
                "output_json": r.output_json,
                "priority": r.priority,
                "flags": r.flags,
            }
            for r in results[:25]
        ],
    }
    (out_dir / "qc_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--icd-tree", type=Path, default=DEFAULT_ICD_TREE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to these output_json / source_file names.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = Config.load()
    auditor = CorpusAuditor(
        config=config,
        cases_dir=args.cases_dir,
        progress_path=args.progress,
        source_dir=args.source_dir,
        icd_tree_path=args.icd_tree,
    )
    only = set(args.only) if args.only else None
    results = auditor.audit(only=only, limit=args.limit)
    write_outputs(results, args.out, config)
    flagged = sum(1 for r in results if r.flags)
    print(f"Audited {len(results)} cases — {flagged} flagged")
    print(f"  qc.csv        -> {args.out / 'qc.csv'}")
    print(f"  qc_report.json -> {args.out / 'qc_report.json'}")


if __name__ == "__main__":
    main()
