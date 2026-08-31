"""
CLI entry point for PSLA experiments.

Usage:
    uv run psla run-experiment 1.1      # Run a specific experiment
    uv run psla extract <file>          # Extract a single case
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from hklii_psla.experiments.experiment import BaseExperiment

from typing import Annotated

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


def main():
    app()


if __name__ == "__main__":
    main()
