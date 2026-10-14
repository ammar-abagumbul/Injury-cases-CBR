"""Run injury severity assessment over a directory of extracted Case JSON files.

Usage:
    uv run python scripts/assess_injury_severity.py <case_dir> [--out <report.json>]
    uv run python scripts/assess_injury_severity.py <case_dir> --llm-model gpt

Each file in <case_dir> must be a full extracted Case dump (metadata,
plaintiff, injuries, treatment, losses, psla, death) — e.g. anything under
output/experiments/*/primary or experiment4/test_list.

Prints a summary table (estimated vs. extracted category, agreement) and
writes the full per-case assessment to --out (default:
"<case_dir>_severity_report.json" as a *sibling* of <case_dir>, not inside
it — so re-running never picks up its own prior output as a case file).

--llm-model is required for a meaningful result: severity is scored from the
treatment narrative classified into ICHI action codes, one LLM call per
case. Without it, the only remaining signal is each injury's own
injury_type flag — composite/category will read as low, but that means
"unassessed", not "confirmed mild".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hklii_psla.severity import assess_case_directory

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", help="Directory of extracted Case *.json files")
    parser.add_argument("--out", default=None, help="Output report path (JSON)")
    parser.add_argument(
        "--icd-depth", type=int, default=2,
        help="ICD-11 truncation depth (1=block e.g. NA00, 2=category e.g. NA00.0)",
    )
    parser.add_argument(
        "--llm-model", default=None,
        help=(
            "Model family to use for reading free-text treatment narrative "
            "(e.g. 'gpt', 'deepseek', 'ollama' — see model_factory.MODEL_FACTORIES). "
            "Effectively required: without it there is almost no severity signal."
        ),
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.is_dir():
        console.print(f"[red]Not a directory: {case_dir}[/red]")
        return 1

    llm = None
    if args.llm_model:
        from hklii_psla.model_factory import create_model

        llm = create_model(args.llm_model)
        console.print(f"[cyan]Using {args.llm_model} for treatment narrative reading[/cyan]")

    assessments = assess_case_directory(case_dir, icd_max_depth=args.icd_depth, llm=llm)
    if not assessments:
        console.print(f"[yellow]No *.json case files found under {case_dir}[/yellow]")
        return 1

    table = Table(title="Injury Severity Assessment")
    table.add_column("Case")
    table.add_column("Composite", justify="right")
    table.add_column("Completeness", justify="right")
    table.add_column("Estimated")
    table.add_column("Extracted")
    table.add_column("Agree?")

    agree_count = 0
    comparable_count = 0
    for a in assessments:
        agree_str = "-"
        if a.agrees_with_extracted is not None:
            comparable_count += 1
            agree_str = "✓" if a.agrees_with_extracted else "✗"
            if a.agrees_with_extracted:
                agree_count += 1
        table.add_row(
            a.case_id,
            f"{a.treatment_complexity.composite:.2f}",
            f"{a.treatment_complexity.data_completeness:.2f}",
            a.estimated_category.value,
            a.extracted_category.value if a.extracted_category else "-",
            agree_str,
        )

    console.print(table)
    if comparable_count:
        console.print(
            f"\n[bold]Agreement with extracted overall_category: "
            f"{agree_count}/{comparable_count}[/bold]"
        )
    if llm is not None:
        no_narrative = sum(1 for a in assessments if a.narrative is None)
        if no_narrative:
            console.print(
                f"[bold yellow]{no_narrative}/{len(assessments)} case(s) have no narrative "
                f"assessment — either the case had no treatment text to read, or the LLM "
                f"call failed (see WARNING lines above for the actual error). Those cases' "
                f"composite/category reflect only injury_type — treat as unassessed, not "
                f"confirmed mild.[/bold yellow]"
            )

    default_out = case_dir.parent / f"{case_dir.name}_severity_report.json"
    out_path = Path(args.out) if args.out else default_out
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in assessments], f, indent=2, ensure_ascii=False)
    console.print(f"[green]Full report written to {out_path}[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
