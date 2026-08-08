"""
CLI entry point for PSLA experiments.

Usage:
    uv run psla run-stage1              # Run all Stage 1 experiments
    uv run psla run-experiment 1.1      # Run a specific experiment
    uv run psla extract <file>          # Extract a single case
    uv run psla validate                # Validate schemas
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hklii_psla.config import settings
from hklii_psla.extractor.single_pass import SinglePassExtractor
from hklii_psla.model_factory import create_model

app = typer.Typer(
    name="psla",
    help="HKLII PSLA Experiments CLI",
    add_completion=False,
)
console = Console()


@app.command()
def run_stage1(
    providers: Optional[str] = typer.Option(
        None, "--providers", help="Comma-separated list of providers, e.g. 'deepseek,glm'"
    ),
):
    """Run all Stage 1 experiments (1.1–1.5)."""
    from hklii_psla.experiments.runner import run_stage1_all

    provider_list = providers.split(",") if providers else None
    run_stage1_all(providers=provider_list)


@app.command()
def run_experiment(
    experiment_id: str = typer.Argument(help="Experiment ID, e.g. '1.1', '1.2'"),
    providers: Optional[str] = typer.Option(
        None, "--providers", help="Comma-separated list of providers"
    ),
):
    """Run a specific experiment."""
    from hklii_psla.experiments.runner import (
        run_experiment_1_1,
        run_experiment_1_2,
        run_experiment_1_3,
        run_experiment_1_5,
        get_case_files,
        save_batch,
    )

    case_files = get_case_files()
    provider = (providers or "deepseek").split(",")[0]

    experiments = {
        "1.1": lambda: run_experiment_1_1(case_files),
        "1.2": lambda: run_experiment_1_2(case_files, provider=provider),
        "1.3": lambda: run_experiment_1_3(case_files, provider=provider),
        "1.5": lambda: run_experiment_1_5(case_files, provider=provider),
    }

    if experiment_id not in experiments:
        console.print(f"[red]Unknown experiment: {experiment_id}[/red]")
        console.print(f"Available: {list(experiments.keys())}")
        raise typer.Exit(1)

    batch = experiments[experiment_id]()
    save_batch(batch, settings.OUTPUT_DIR / "experiments")


@app.command()
def extract(
    file: Path = typer.Argument(help="Path to judgment text file"),
    provider: str = typer.Option("deepseek", help="Model provider"),
    strategy: str = typer.Option("single-pass", help="Extraction strategy"),
    output: Optional[Path] = typer.Option(None, help="Output JSON file path"),
):
    """Extract structured features from a single judgment."""
    llm = create_model(provider)
    judgment = SinglePassExtractor.load_judgment(file)

    from hklii_psla.model_factory import MODEL_PROVIDER_NAMES
    model_name = MODEL_PROVIDER_NAMES[provider]

    if strategy == "single-pass":
        extractor = SinglePassExtractor(llm, model_name=model_name)
    elif strategy == "section-by-section":
        from hklii_psla.extractor.section_by_section import SectionBySectionExtractor
        extractor = SectionBySectionExtractor(llm, model_name=model_name)
    elif strategy == "multi-agent":
        from hklii_psla.extractor.multi_agent import MultiAgentExtractor
        extractor = MultiAgentExtractor(llm, model_name=model_name)
    else:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        raise typer.Exit(1)

    console.print(f"Extracting from [bold]{file.name}[/bold] using {provider}/{strategy}...")
    result = extractor.extract(judgment)

    if result.success:
        console.print(f"[green]Extraction successful[/green] ({result.duration_ms:.0f}ms)")
        case_json = result.case.model_dump(mode="json") if result.case else {}

        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(case_json, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"Saved to [bold]{output}[/bold]")
        else:
            console.print_json(json.dumps(case_json, indent=2, ensure_ascii=False, default=str))
    else:
        console.print(f"[red]Extraction failed: {result.error}[/red]")


@app.command()
def validate_schemas():
    """Validate that Pydantic schemas can be instantiated correctly."""
    from hklii_psla.schemas import (
        Case,
        CaseMetadata,
        PlaintiffBackground,
        Gender,
        Injury,
        InjurySummary,
        InjuryType,
        InjuryCategory,
        Treatment,
        LossSummary,
        PSLAAward,
        InjuryLossRelation,
        ALL_LOSS_CATEGORIES,
    )

    console.print("[bold]Schema Validation[/bold]")

    # Test basic instantiation
    try:
        _case = Case(
            metadata=CaseMetadata(
                neutral_citation="[2020] HKDC 1745",
                action_number="DCPI 2723/2018",
                case_name="TEST v. TEST",
            ),
            plaintiff=PlaintiffBackground(
                gender=Gender.MALE,
                age_at_accident=45,
            ),
            injuries=InjurySummary(
                injuries=[
                    Injury(
                        description="Fractured left leg",
                        injury_type=InjuryType.TEMPORARY,
                        body_part="left leg",
                    )
                ],
                overall_category=InjuryCategory.SERIOUS,
            ),
            treatment=Treatment(
                hospitalisation_days=14,
                operations_count=1,
            ),
            losses=LossSummary(),
            psla=PSLAAward(amount=500000.0),
            injury_loss_relations=[
                InjuryLossRelation(
                    injury="Fractured left leg",
                    loss="Loss of mobility",
                    evidence=["paragraph 15"],
                )
            ],
        )
        console.print("[green]✓[/green] Case instantiation OK")
        console.print(f"  Loss categories available: {len(ALL_LOSS_CATEGORIES)}")
    except Exception as e:
        console.print(f"[red]✗[/red] Validation failed: {e}")
        raise typer.Exit(1)

    console.print("[green]All schema validations passed.[/green]")


@app.command()
def list_providers():
    """List available model providers."""
    from hklii_psla.model_factory import MODEL_PROVIDER_NAMES

    table = Table(title="Available Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("API Key Env", style="yellow")

    env_vars = {
        "deepseek": "OPENROUTER_API_KEY",
        "qwen": "OPENROUTER_API_KEY",
        "glm": "OPENROUTER_API_KEY",
        "kimi": "OPENROUTER_API_KEY",
        "minimax": "OPENROUTER_API_KEY",
        "ollama": "(local)",
    }

    for provider, model in MODEL_PROVIDER_NAMES.items():
        table.add_row(provider, model, env_vars.get(provider, ""))

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
