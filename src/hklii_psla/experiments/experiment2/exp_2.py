import logging
import json
import random
import uuid

from rich.console import Console

from hklii_psla.experiments.experiment import (
    BaseExperiment,
    ExperimentBatch,
    ExperimentRun,
    ExtractionResult
)
from hklii_psla.experiments.experiment1.reconciler import (
    Reconciler,
    ReconcilerConfig,
    ReconciliationField
)
from hklii_psla.extractor.single_pass import SinglePassExtractor
from hklii_psla.model_factory import create_model
from hklii_psla.schemas import Case

from pathlib import Path
from pydantic import BaseModel
from typing import ClassVar, override

logger = logging.getLogger(__name__)
console = Console()

class Exp2Config(BaseModel):
    id: str
    model: str
    provider: str
    num_runs: int
    sample_size: int
    permutate: bool
    case_files_path: str
    eval_schema_path: str
    output_dir: str
    seed: int | None = 42


class ReconcilationExperiment(BaseExperiment[Exp2Config], experiment_id="reconciliation"):

    _config: Exp2Config
    _config_model: ClassVar[type[BaseModel]] = Exp2Config


    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.batch: ExperimentBatch = ExperimentBatch(experiment_id=self._experiment_id)
        self.case_files: list[Path] = self.fetch_case_files()
        random.seed(self._config.seed)

    def fetch_case_files(self) -> list[Path]:
        case_dir: str = self._config.case_files_path
        return list(Path(case_dir).glob("*"))

    @override
    def run(self) -> None:

        llm = create_model(self._config.model)
        extractor = SinglePassExtractor(llm, model_name=self._config.model)
        reconciation_fields = [
            ReconciliationField(name="injuries", dependencies=[]),
            ReconciliationField(name="losses", dependencies=["injuries"]),
        ]
        reconciler_config = ReconcilerConfig(
            fields=reconciation_fields,
            schema=Case,
            additional_instructions=""
        )
        reconciler = Reconciler(
            model=llm,
            config=reconciler_config
        )

        for batch_idx in range(self._config.num_runs):
            if self._config.permutate:
                random.shuffle(self.case_files)

            for file in self.case_files:
                results: list[ExtractionResult] = []
                for _ in range(self._config.sample_size):
                    judgement_text = extractor.load_judgment(file)
                    result = extractor.extract(judgement_text, prompt_style="zero-shot")
                    results.append(result)

                try:
                    reconciled_result = reconciler.reconcile(results)
                    reconciled_result.duration_ms += sum([r.duration_ms for r in results])
                    reconciled_result.token_count = sum(
                        [r.token_count for r in results],
                        reconciled_result.token_count
                    )
                    filename = self._generate_filename(reconciled_result)
                except Exception as e:
                    logger.error(f"    error — {e}")
                    continue

                run = ExperimentRun(
                    provider=self._config.provider,
                    model=self._config.model,
                    strategy="single-pass",
                    prompt_style="zero-shot",
                    case_file=str(file),
                    result=reconciled_result,
                    filename=filename,
                )

                self.batch.runs.append(run)

                if reconciled_result.success:
                    logger.info(f"    ok — {reconciled_result.duration_ms:.0f}ms, {reconciled_result.token_count.total_tokens} tokens")
                else:
                    logger.info(f"    fail — {reconciled_result.error}")
                    logger.error(f"    debug — {reconciled_result.debug}")

            _ = self.save_batch(batch_index=batch_idx)
            self.batch.runs.clear()


    def save_batch(self, batch_index: int) -> Path:
        """Save extracted cases for a single run batch to disk."""
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_dir = output_dir / f"batch_{batch_index}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        for r in self.batch.runs:
            if r.result and r.result.case:
                case_json = r.result.case.model_dump()
                filepath = batch_dir / (r.filename or f"{uuid.uuid4()}.json")
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(case_json, f, indent=2, ensure_ascii=False)

        return batch_dir

    @override
    def save_results(self) -> None:
        """Save anomaly tables and experiment summary to the output directory."""
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold green]Results saved to {output_dir}[/bold green]")

    def _generate_filename(self, result: ExtractionResult) -> str:
        if result.case and result.case.metadata and result.case.metadata.neutral_citation:
            return f"{result.case.metadata.neutral_citation}.json"
        return f"{uuid.uuid4()}.json"
