"""
Experiment runner for Stage 1: Reliable Structured Feature Extraction.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path

from hklii_psla.config import settings
from hklii_psla.extractor.base import ExtractionResult
from hklii_psla.extractor.single_pass import SinglePassExtractor
from hklii_psla.model_factory import create_model, MODEL_PROVIDER_NAMES

from rich.console import Console
from rich.panel import Panel

from typing import Any

console = Console()

@dataclass
class ExperimentRun:
    """Record of a single extraction run."""

    provider: str
    model: str
    strategy: str  # "single-pass", "section-by-section", "multi-agent"
    prompt_style: str  # "zero-shot" or "few-shot"
    case_file: str
    result: ExtractionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "strategy": self.strategy,
            "prompt_style": self.prompt_style,
            "case_file": self.case_file,
        }
        if self.result:
            d.update({
                "success": self.result.success,
                "duration_ms": self.result.duration_ms,
                "token_count": self.result.token_count.to_dict(),
                "error": self.result.error,
                "case_json": self.result.case.model_dump(mode="json") if self.result.case else None,
            })
        return d


@dataclass
class ExperimentBatch:
    """Collection of runs for a single experiment."""

    experiment_id: str
    runs: list[ExperimentRun] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        total = len(self.runs)
        successful = sum(1 for r in self.runs if r.result and r.result.success)
        total_duration = sum(r.result.duration_ms for r in self.runs if r.result)
        total_tokens = sum(r.result.token_count.total_tokens for r in self.runs if r.result)
        return {
            "experiment_id": self.experiment_id,
            "total_runs": total,
            "successful": successful,
            "failed": total - successful,
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
        }


def run_experiment_1_1(case_files: list[Path]) -> ExperimentBatch:
    batch = ExperimentBatch(experiment_id="1.1-model-comparison")

    for provider, model_name in MODEL_PROVIDER_NAMES.items():
        print(f"\n{'='*60}")
        print(f"Experiment 1.1 — Provider: {provider} ({model_name})")
        print(f"{'='*60}")

        try:
            llm = create_model(provider)
        except Exception as e:
            print(f"  SKIP: Failed to create model for {provider}: {e}")
            continue

        extractor = SinglePassExtractor(llm, model_name=model_name)

        for case_file in case_files:
            print(f"  **Processing**: {case_file.name}")
            judgment = SinglePassExtractor.load_judgment(case_file)

            result = extractor.extract(judgment, prompt_style="zero-shot")
            run = ExperimentRun(
                provider=provider,
                model=model_name,
                strategy="single-pass",
                prompt_style="zero-shot",
                case_file=str(case_file.name),
                result=result,
            )
            batch.runs.append(run)


            if result.success:
                console.print(f"    OK — {result.duration_ms:.0f}ms, {result.token_count.total_tokens} tokens")
            else:
                console.print(Panel(f"    FAIL — {result.error}"))

    return batch


def get_case_files() -> list[Path]:
    """Get all sample case files."""
    sample_dir = settings.SAMPLE_DIR
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    return sorted(sample_dir.glob("*.txt"))


def save_batch(batch: ExperimentBatch, output_dir: Path):
    """Save experiment batch results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{batch.experiment_id}.json"

    data = {
        "experiment_id": batch.experiment_id,
        "summary": batch.summary(),
        "runs": [
            {
                **r.to_dict(),
                # Don't duplicate case_json in the run record if it's large;
                # we store it separately
                "case_json": None,  # stored per-case separately
            }
            for r in batch.runs
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Saved batch to {filepath}")

    # Save individual case extractions
    cases_dir = output_dir / f"{batch.experiment_id}_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for run in batch.runs:
        if run.result and run.result.case:
            safe_name = run.case_file.replace(".txt", "").replace("/", "_")
            case_path = cases_dir / f"{safe_name}_{run.provider}_{run.strategy}.json"
            with open(case_path, "w", encoding="utf-8") as f:
                json.dump(
                    run.result.case.model_dump(mode="json"),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
