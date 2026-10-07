"""
CLI entry point for PSLA experiments.

Usage:
    uv run psla run-experiment 1.1      # Run a specific experiment
    uv run psla extract <file>          # Extract a single case
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from hklii_psla.experiments.experiment import BaseExperiment

logging.basicConfig(level=logging.INFO)


app = typer.Typer(
    name="psla",
    help="HKLII PSLA Experiments CLI",
    add_completion=False,
)
console = Console()


@app.command()
def run_experiment(
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID, e.g. '1.1', '1.2'")],
    config_file: Annotated[str, typer.Argument(help="Path to the config file")],
):
    """Run a specific experiment."""

    config_path = Path(config_file)

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    experiment_cls = BaseExperiment.registry.get(experiment_id)
    if experiment_cls is None:
        console.print(f"[red]Experiment class not found for key: {experiment_id}[/red]")
        console.print(f"Registered experiments: {list(BaseExperiment.registry.keys())}")
        raise typer.Exit(1)

    console.print(f"[bold]Running experiment {experiment_id} ({experiment_id})...[/bold]")

    experiment = experiment_cls(config_path)

    experiment.run()
    experiment.save_results()

    console.print(f"[bold green]Experiment {experiment_id} complete.[/bold green]")


@app.command()
def placeholder():
    """Keeps the app multi-command."""


@app.command(name="audit-corpus")
def audit_corpus(
    out: Annotated[
        str,
        typer.Option(help="Output directory for qc.csv / qc_report.json"),
    ] = "output/experiments/corpus_extraction/qc",
    limit: Annotated[
        int | None,
        typer.Option(help="Limit the number of cases audited"),
    ] = None,
):
    """Run the deterministic QC audit over the extracted corpus."""
    from hklii_psla.experiments.experiment6.audit import main as audit_main

    argv = ["--out", out]
    if limit is not None:
        argv += ["--limit", str(limit)]
    audit_main(argv)


@app.command(name="patch-corpus")
def patch_corpus(
    ops: Annotated[
        str,
        typer.Option(help="Comma-separated operations: citation,psych,coarse"),
    ] = "citation,psych",
    select: Annotated[
        str,
        typer.Option(help="Comma-separated QC flags selecting cases (empty = all)"),
    ] = "",
    limit: Annotated[
        int | None,
        typer.Option(help="Limit the number of cases processed"),
    ] = None,
    provider: Annotated[
        str,
        typer.Option(help="Model provider for the coarse re-walk"),
    ] = "gpt",
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Write the validated patches (with backups)"),
    ] = False,
    show: Annotated[
        int,
        typer.Option(help="Print the first N proposed patches"),
    ] = 0,
):
    """Build (and optionally apply) minimal partial-edit patches."""
    from hklii_psla.experiments.experiment6.patcher import main as patch_main

    argv = ["--ops", ops, "--provider", provider, "--show", str(show)]
    if select:
        argv += ["--select", select]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if apply:
        argv.append("--apply")
    patch_main(argv)


def main():
    app()


if __name__ == "__main__":
    main()
