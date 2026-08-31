import itertools
import json
import logging
import random
from time import process_time
import uuid

from pathlib import Path
from typing import Any, ClassVar, override

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from hklii_psla.experiments.experiment import BaseExperiment, ExperimentBatch, ExperimentRun
from hklii_psla.extractor.base import ExtractionResult
from hklii_psla.extractor.single_pass import SinglePassExtractor
from hklii_psla.model_factory import create_model, MODEL_PROVIDER_NAMES

from hklii_psla.extract_bench import ReportBuilder, ReportConfig

logger = logging.getLogger(__name__)
console = Console()


class Exp1Config(BaseModel):
    id: str
    model: str
    provider: str
    num_runs: int
    permutate: bool
    case_files_path: str
    eval_schema_path: str
    output_dir: str
    seed: int | None = 42


class ConsistencyExperiment(BaseExperiment[Exp1Config], experiment_id="model-consistency"):

    _config: Exp1Config
    _config_model: ClassVar[type[BaseModel]] = Exp1Config

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.batch: ExperimentBatch = ExperimentBatch(experiment_id=self._experiment_id)
        self.anomalies: dict[int, dict[str, list[str]]] = {} # run_index -> {filename -> [anomaly descriptions]}
        try:
            with Path(self._config.eval_schema_path).open() as f:
                self.schema_json: dict[str, Any] = json.load(f)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error loading schema: {e}")

    @override
    def run(self) -> None:
        # try:
        #     output_dir = Path(self._config.output_dir)
        #     processed_batches = [
        #         (output_dir / "batch_0", 0),
        #         (output_dir / "batch_1", 1),
        #     ]
        #     console.print("[bold yellow]Evaluating experiment ...[/bold yellow]")
        #     self.cross_evaluate(processed_batches)
        # except Exception as e:
        #     console.print(f"[bold red]Error building report: {e}[/bold red]")

        # return

        random.seed(self._config.seed)

        processed_batches: list[tuple[Path, int]] = []

        if self._config.model not in MODEL_PROVIDER_NAMES:
            raise ValueError(f"Invalid model: {self._config.model}")

        try:
            llm = create_model(self._config.model)
            case_files_glob = Path(self._config.case_files_path).glob("*.txt")
            case_files = list(case_files_glob)
            extractor = SinglePassExtractor(llm, model_name=self._config.model)

            for i in range(self._config.num_runs):
                if self._config.permutate:
                    random.shuffle(case_files)

                for file in case_files:
                    judgment_text = SinglePassExtractor.load_judgment(file)
                    result = extractor.extract(judgment_text, prompt_style="zero-shot")
                    filename = self._generate_filename(result)

                    run = ExperimentRun(
                        provider=self._config.provider,
                        model=self._config.model,
                        strategy="single-pass",
                        prompt_style="zero-shot",
                        case_file=str(file),
                        result=result,
                        filename=filename,
                    )

                    self.batch.runs.append(run)
                    self.check_anomalies(result, filename, run_index=i)

                    if result.success:
                        logger.info(f"    OK — {result.duration_ms:.0f}ms, {result.token_count.total_tokens} tokens")
                    else:
                        logger.info(f"    FAIL — {result.error}")
                        logger.error(f"    DEBUG — {result.debug}")

                batch_idx = i
                batch_path = self.save_batch(run_index=i)
                processed_batches.append((batch_path, batch_idx))
                self.batch.runs.clear()

            try:
                console.print("[bold yellow]Evaluating experiment ...[/bold yellow]")
                self.cross_evaluate(processed_batches)
            except Exception as e:
                console.print(f"[bold red]Error building report: {e}[/bold red]")


        except Exception as e:
            raise RuntimeError(f"Error occured while running experiment: {self._config.model}\n\n{e}")

    def cross_evaluate(self, processed_batches: list[tuple[Path, int]]) -> None:
        for batch in itertools.combinations(processed_batches, 2):
            batch_a, batch_b = batch
            for file_a in batch_a[0].glob("*.json"):
                file_b = batch_b[0] / file_a.name

                if not file_b.exists():
                    continue

                batch_a_json: dict[str, Any] = json.load(file_a.open())
                batch_b_json: dict[str, Any] = json.load(file_b.open())

                config = ReportConfig(
                    output_dir=Path(self._config.output_dir),
                    output_name=f"{batch_a[1]}_{batch_b[1]}",
                    max_reasoning_length=2000,
                )

                builder = ReportBuilder(config)
                report = builder.build(self.schema_json, batch_a_json, batch_b_json)
                _ = builder.save(report)

        console.print("[bold green]Completed evaluating experiment ...[/bold green]")

    def check_anomalies(self, result: ExtractionResult, filename: str, run_index: int) -> None:
        """
        Inspect an extraction result for common anomalies and record them.

        Checks performed:
          - Extraction failure (no Case produced)
          - Missing neutral citation
          - Empty comparable PSLA cases
          - No losses extracted
          - Injuries with both icd_code and icd_description set to None
        """
        if run_index not in self.anomalies:
            self.anomalies[run_index] = {}

        issues: list[str] = []

        if result.case is None:
            issues.append("Extraction failed (no Case produced)")
            self.anomalies[run_index][filename] = issues
            return

        case = result.case

        # 1. Missing neutral citation
        if not case.metadata.neutral_citation:
            issues.append("Missing neutral citation")

        # 2. Empty comparable PSLA cases
        if not case.psla.comparable_cases:
            issues.append("No comparable PSLA cases")

        # 3. No losses extracted
        if not case.losses.losses:
            issues.append("No losses extracted")

        # 4. Injuries missing ICD codes (both code and description are None)
        injuries_missing_icd = [
            inj.injury_id
            for inj in case.injuries.injuries
            if inj.icd_code is None and inj.icd_description is None
        ]
        if injuries_missing_icd:
            issues.append(
                f"Injuries missing ICD codes: {', '.join(injuries_missing_icd)}"
            )

        if issues:
            self.anomalies[run_index][filename] = issues

    def save_batch(self, run_index: int) -> Path:
        """Save extracted cases for a single run batch to disk."""
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_dir = output_dir / f"batch_{run_index}"
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

        anomalies_file = output_dir / "anomalies.json"
        with open(anomalies_file, "w", encoding="utf-8") as f:
            json.dump(self.anomalies, f, indent=2, ensure_ascii=False)

        total_anomalous_files = 0
        for run_index in sorted(self.anomalies.keys()):
            run_anomalies = self.anomalies[run_index]

            if not run_anomalies:
                console.print(f"[green]Run {run_index}: No anomalies detected.[/green]")
                continue

            total_anomalous_files += len(run_anomalies)

            table = Table(title=f"Anomalies — Run {run_index}")
            table.add_column("File", style="cyan", no_wrap=True)
            table.add_column("Anomalies", style="yellow")

            for filename, issues in run_anomalies.items():
                table.add_row(filename, "\n".join(issues))

            console.print(table)

        console.print(f"\n[bold green]Results saved to {output_dir}[/bold green]")

    def _generate_filename(self, result: ExtractionResult) -> str:
        if result.case and result.case.metadata and result.case.metadata.neutral_citation:
            return f"{result.case.metadata.neutral_citation}.json"
        return f"{uuid.uuid4()}.json"
