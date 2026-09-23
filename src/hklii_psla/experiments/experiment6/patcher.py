"""Partial-edit agent for the Experiment 6 corpus clean-up.

This module builds **minimal, reviewable patches** for individual ``Case``
JSONs produced by Experiment 5. It never re-extracts a whole case and never
renumbers ``inj_00x`` ids.

Operations
----------
``citation``
    Deterministic: rewrite ``metadata.neutral_citation`` extraction-first —
    prefer a citation recovered from the already-extracted field, falling back
    to the source header only when that fails; an extraction/source
    disagreement is left for review.  Also fills an empty ``action_number``
    from the source header.
``psych``
    Deterministic: if psychiatric content is detected (bag of words over
    injuries + manifestations) and no ``Personality change`` loss exists,
    append exactly one loss whose description is taken verbatim from the
    detected item's ``source`` quotes.
``coarse``
    LLM: re-walk the ICD-11 tree for injuries carrying a coarse / prematurely
    stopped code, forbidding the coarse denylist and forcing one more level.

Every proposed patch is validated with :meth:`Case.model_validate` before it
can be applied. ``apply`` writes a timestamped-agnostic backup under
``cases_backup/`` the first time a file is modified.

CLI::

    uv run psla patch-corpus --ops citation,psych --select \
        CITATION_MISMATCH,PSYCH_NO_PERSONALITY_CHANGE_LOSS --apply
    uv run psla patch-corpus --ops coarse --provider gpt --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hklii_psla.extractor.icd_classifier import (
    areclassify_injury,
    build_icd_index,
    load_icd_tree,
)
from hklii_psla.schemas import ALL_LOSS_CATEGORIES, Case

from .audit import (
    DEFAULT_COURTS,
    REPO_ROOT,
    Config,
    _make_matcher,
    make_citation_recovery_re,
    resolve_citation,
)

DEFAULT_CASES_DIR = REPO_ROOT / "output/experiments/corpus_extraction/cases"
DEFAULT_SOURCE_DIR = REPO_ROOT / "old_pi_cases_web_text"
DEFAULT_ICD_TREE = REPO_ROOT / "data/ICD-11.json"
DEFAULT_PROGRESS = REPO_ROOT / "output/experiments/corpus_extraction/progress.csv"
DEFAULT_PATCH_DIR = REPO_ROOT / "output/experiments/corpus_extraction/patches"
DEFAULT_BACKUP_DIR = REPO_ROOT / "output/experiments/corpus_extraction/cases_backup"


# ---------------------------------------------------------------------------
# Patch representation
# ---------------------------------------------------------------------------


@dataclass
class Change:
    path: str
    before: Any
    after: Any
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "note": self.note,
        }


@dataclass
class Patch:
    output_json: str
    case_path: Path
    ops: list[str]
    changes: list[Change] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    valid: bool = False
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_json": self.output_json,
            "case_path": str(self.case_path),
            "ops": self.ops,
            "valid": self.valid,
            "error": self.error,
            "warnings": self.warnings,
            "changes": [c.to_dict() for c in self.changes],
        }

    def render(self) -> str:
        lines = [f"# {self.output_json}  ops={','.join(self.ops)}  valid={self.valid}"]
        if self.error:
            lines.append(f"!! {self.error}")
        for warning in self.warnings:
            lines.append(f"~ {warning}")
        for change in self.changes:
            lines.append(f"  {change.path}")
            lines.append(f"    - {_short(change.before)}")
            lines.append(f"    + {_short(change.after)}")
            if change.note:
                lines.append(f"    # {change.note}")
        if self.is_empty:
            lines.append("  (no change)")
        return "\n".join(lines)


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + " …"


# ---------------------------------------------------------------------------
# Deterministic operations
# ---------------------------------------------------------------------------


def fix_citation(
    case: dict[str, Any],
    source_text: str,
    pattern: re.Pattern[str] | None = None,
) -> list[Change]:
    """Rewrite ``metadata.neutral_citation``, extraction-first.

    The already-extracted field is preferred whenever it contains a known
    neutral citation (stripping any surrounding text); the source header is
    only consulted when the extraction yields nothing.  An extraction/source
    disagreement is left untouched and surfaced by the audit as
    ``CITATION_CONFLICT``.
    """
    pattern = pattern or make_citation_recovery_re(list(DEFAULT_COURTS))
    meta = case.setdefault("metadata", {})
    current = (meta.get("neutral_citation") or "").strip()

    resolution = resolve_citation(current, source_text, pattern)
    if resolution.outcome == "conflict":
        return []

    changes: list[Change] = []
    if resolution.resolved and resolution.resolved != current:
        note = (
            "recovered from extracted field"
            if resolution.outcome == "extra_text"
            else "recovered from source header"
        )
        changes.append(
            Change(
                path="metadata.neutral_citation",
                before=current,
                after=resolution.resolved,
                note=note,
            )
        )
        meta["neutral_citation"] = resolution.resolved

    if resolution.source_action and not (meta.get("action_number") or "").strip():
        changes.append(
            Change(
                path="metadata.action_number",
                before=meta.get("action_number") or "",
                after=resolution.source_action,
                note="filled from source header",
            )
        )
        meta["action_number"] = resolution.source_action

    return changes


def _psych_items(
    case: dict[str, Any], matcher: Callable[[str | None], str | None]
) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    injuries = case.get("injuries", {})
    for injury in injuries.get("injuries", []) or []:
        text = (injury.get("description") or "") + " " + " ".join(
            injury.get("source") or []
        )
        if matcher(text):
            items.append(("injury", injury))
    for manifest in injuries.get("manifestations", []) or []:
        if matcher(manifest.get("description")):
            items.append(("manifestation", manifest))
    return items


def add_personality_change_loss(
    case: dict[str, Any], config: Config
) -> list[Change]:
    """Append one ``Personality change`` loss if psych content is detected.

    The original (possibly null-ICD) injury entry is left untouched so injury
    ids are never renumbered.
    """
    matcher = _make_matcher(
        config.psych["keywords"], config.psych.get("exclusions", [])
    )
    target = config.psych["target_loss_category"]
    if target not in ALL_LOSS_CATEGORIES:
        raise ValueError(f"unknown target loss category: {target}")

    loss_summary = case.setdefault("losses", {}).setdefault("losses", [])
    if any((loss.get("category") == target) for loss in loss_summary):
        return []

    items = _psych_items(case, matcher)
    if not items:
        return []

    injury_quotes: list[str] = []
    descriptions: list[str] = []
    primary_injury_id: str | None = None
    for kind, item in items:
        if kind == "injury":
            quotes = [q for q in (item.get("source") or []) if q.strip()]
            injury_quotes.extend(quotes)
            if quotes and primary_injury_id is None:
                primary_injury_id = item.get("injury_id")
            if item.get("description"):
                descriptions.append(item["description"])
        else:
            if item.get("description"):
                descriptions.append(item["description"])

    if config.psych.get("description_from") == "source" and injury_quotes:
        description = " ".join(injury_quotes)
    else:
        description = " ".join(descriptions)

    if not primary_injury_id:
        linked = next(
            (
                item.get("caused_by_injury_id")
                for kind, item in items
                if kind == "manifestation" and item.get("caused_by_injury_id")
            ),
            None,
        )
        primary_injury_id = linked or ""

    loss = {
        "category": target,
        "description": description,
        "caused_by_injury_id": primary_injury_id,
    }
    loss_summary.append(loss)
    return [
        Change(
            path=f"losses.losses[{len(loss_summary) - 1}]",
            before=None,
            after=loss,
            note="psychiatric content detected; one loss per case",
        )
    ]


# ---------------------------------------------------------------------------
# Coarse-node re-walk (LLM)
# ---------------------------------------------------------------------------


def _has_allowed_children(node: dict[str, Any], forbidden: set[str]) -> bool:
    return any(
        child.get("code") not in forbidden for child in node.get("children", [])
    )


def choose_rewalk_start(
    code: str,
    by_code: dict[str, dict[str, Any]],
    parent_of: dict[str, str | None],
    forbidden: set[str],
) -> dict[str, Any] | None:
    """Pick the node from which to re-walk a coarse / premature code.

    If the code itself still has allowed children, start there (one more level
    is forced). Otherwise walk up to the nearest ancestor that offers a
    non-forbidden option.
    """
    node = by_code.get(code)
    if node is not None and _has_allowed_children(node, forbidden):
        return node

    parent_code = parent_of.get(code)
    while parent_code:
        parent = by_code[parent_code]
        if _has_allowed_children(parent, forbidden):
            return parent
        parent_code = parent_of.get(parent_code)
    return None


async def reclassify_coarse_injuries(
    case: dict[str, Any],
    model: Any,
    by_code: dict[str, dict[str, Any]],
    parent_of: dict[str, str | None],
    coarse_codes: set[str],
    which: str = "all",
) -> list[Change]:
    """Re-walk each qualifying injury and overwrite only its ICD fields."""
    from hklii_psla.schemas import Injury

    injuries = case.get("injuries", {}).get("injuries", []) or []
    changes: list[Change] = []

    for index, injury in enumerate(injuries):
        code = injury.get("icd_code")
        if not code or code not in by_code:
            continue
        node = by_code[code]
        is_coarse = code in coarse_codes
        is_stopped_early = bool(node.get("children"))
        if which == "coarse" and not is_coarse:
            continue
        if which == "stopped_early" and not is_stopped_early:
            continue
        if which == "all" and not (is_coarse or is_stopped_early):
            continue

        start = choose_rewalk_start(
            code, by_code, parent_of, coarse_codes
        )
        if start is None:
            continue

        injury_obj = Injury(
            injury_id=injury.get("injury_id", f"inj_{index:03d}"),
            description=injury.get("description", ""),
            injury_type=injury.get("injury_type", "Temporary injury"),
            body_part=injury.get("body_part"),
            laterality=injury.get("laterality"),
            source=injury.get("source") or [],
            icd_code=code,
            icd_description=injury.get("icd_description"),
        )
        new_injury, _ = await areclassify_injury(
            model,
            injury_obj,
            start,
            forbidden_codes=coarse_codes,
            force_deep=True,
        )
        new_code = new_injury.icd_code
        if not new_code or new_code == code:
            continue

        changes.append(
            Change(
                path=f"injuries.injuries[{index}].icd_code",
                before=code,
                after=new_code,
                note="re-walked from coarse/premature node",
            )
        )
        changes.append(
            Change(
                path=f"injuries.injuries[{index}].icd_description",
                before=injury.get("icd_description"),
                after=new_injury.icd_description,
                note="",
            )
        )
        injury["icd_code"] = new_code
        injury["icd_description"] = new_injury.icd_description

    return changes


# ---------------------------------------------------------------------------
# Proposing / validating / applying
# ---------------------------------------------------------------------------


def validate_case(case: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        Case.model_validate(case)
        return True, None
    except ValidationError as exc:
        return False, str(exc)


def read_source(source_dir: Path, source_file: str) -> str:
    path = source_dir / source_file
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def propose_patch(
    output_json: str,
    case_path: Path,
    source_text: str,
    ops: list[str],
    config: Config,
    model: Any | None = None,
    by_code: dict[str, dict[str, Any]] | None = None,
    parent_of: dict[str, str | None] | None = None,
) -> Patch:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    patch = Patch(output_json=output_json, case_path=case_path, ops=list(ops))

    try:
        if "citation" in ops:
            courts = (
                config.qc["citation"].get("court_allowlist") or list(DEFAULT_COURTS)
            )
            patch.changes.extend(
                fix_citation(case, source_text, make_citation_recovery_re(courts))
            )
        if "psych" in ops:
            patch.changes.extend(add_personality_change_loss(case, config))
        if "coarse" in ops:
            if model is None or by_code is None or parent_of is None:
                raise ValueError("coarse op requires a model and ICD index")
            changes = asyncio.run(
                reclassify_coarse_injuries(
                    case,
                    model,
                    by_code,
                    parent_of,
                    set(config.coarse["nodes"]),
                )
            )
            patch.changes.extend(changes)
    except Exception as exc:  # noqa: BLE001
        patch.error = f"{type(exc).__name__}: {exc}"
        patch.valid = False
        return patch

    if patch.is_empty:
        patch.valid = True
        return patch

    valid, error = validate_case(case)
    patch.valid = valid
    patch.error = error
    if not valid:
        return patch

    # keep the validated (but not model_dumped) case for apply
    patch._case = case  # type: ignore[attr-defined]
    return patch


def apply_patch(patch: Patch, backup_dir: Path) -> bool:
    if not patch.valid or patch.is_empty:
        return False
    case = getattr(patch, "_case", None)
    if case is None:
        return False

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / patch.case_path.name
    if not backup_path.exists():
        shutil.copy2(patch.case_path, backup_path)

    patch.case_path.write_text(
        json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return True


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------


def select_cases(
    qc_csv: Path,
    flags: set[str],
    limit: int | None = None,
    require_all: bool = False,
) -> list[tuple[str, str]]:
    """Return ``(source_file, output_json)`` rows matching the requested flags."""
    import csv

    rows: list[tuple[str, str]] = []
    with qc_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_flags = set(row["flags"].split(";")) if row["flags"] else set()
            if not flags:
                matched = True
            elif require_all:
                matched = flags <= row_flags
            else:
                matched = bool(flags & row_flags)
            if matched:
                rows.append((row["source_file"], row["output_json"]))
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_patch_batch(
    ops: list[str],
    select_flags: set[str],
    qc_csv: Path,
    cases_dir: Path,
    source_dir: Path,
    icd_tree_path: Path,
    patch_dir: Path,
    apply: bool,
    backup_dir: Path,
    limit: int | None = None,
    provider: str = "gpt",
    config: Config | None = None,
) -> list[Patch]:
    config = config or Config.load()
    rows = select_cases(qc_csv, select_flags, limit=limit)
    patch_dir.mkdir(parents=True, exist_ok=True)
    ops_tag = "-".join(sorted(ops)) or "noop"

    model = None
    by_code = parent_of = None
    if "coarse" in ops:
        from hklii_psla.model_factory import create_model

        model = create_model(provider)
        by_code, parent_of = build_icd_index(load_icd_tree(icd_tree_path))

    patches: list[Patch] = []
    for source_file, output_json in rows:
        case_path = cases_dir / output_json
        if not case_path.exists():
            continue
        patch = propose_patch(
            output_json,
            case_path,
            read_source(source_dir, source_file),
            ops,
            config,
            model=model,
            by_code=by_code,
            parent_of=parent_of,
        )
        patches.append(patch)

        proposal_path = patch_dir / f"{Path(output_json).stem}.{ops_tag}.patch.json"
        proposal_path.write_text(
            json.dumps(patch.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if apply and patch.valid and not patch.is_empty:
            apply_patch(patch, backup_dir)

    return patches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 6 partial-edit agent")
    parser.add_argument("--ops", default="citation,psych", help="citation,psych,coarse")
    parser.add_argument(
        "--select",
        default="",
        help="Comma-separated QC flags to select cases (empty = all)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qc-csv", type=Path, default=DEFAULT_PATCH_DIR.parent / "qc/qc.csv")
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--icd-tree", type=Path, default=DEFAULT_ICD_TREE)
    parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--provider", default="gpt")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--show", type=int, default=0, help="Print the first N patches to stdout"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ops = [op.strip() for op in args.ops.split(",") if op.strip()]
    flags = {f.strip() for f in args.select.split(",") if f.strip()}

    patches = run_patch_batch(
        ops=ops,
        select_flags=flags,
        qc_csv=args.qc_csv,
        cases_dir=args.cases_dir,
        source_dir=args.source_dir,
        icd_tree_path=args.icd_tree,
        patch_dir=args.patch_dir,
        apply=args.apply,
        backup_dir=args.backup_dir,
        limit=args.limit,
        provider=args.provider,
    )

    with_changes = [p for p in patches if not p.is_empty]
    invalid = [p for p in patches if not p.valid]
    applied = sum(1 for p in patches if args.apply and p.valid and not p.is_empty)
    print(
        f"Proposed {len(patches)} patches — {len(with_changes)} with changes, "
        f"{len(invalid)} invalid, {applied} applied"
    )
    print(f"  proposals -> {args.patch_dir}")
    if args.apply:
        print(f"  backups   -> {args.backup_dir}")

    for patch in patches[: args.show]:
        print()
        print(patch.render())


if __name__ == "__main__":
    main()
