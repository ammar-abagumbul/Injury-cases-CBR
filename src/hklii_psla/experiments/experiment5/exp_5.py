"""Experiment 5 — asynchronous corpus extraction.

Run the async single-pass extractor over a large corpus of judgment ``.txt``
files.  Concurrency is bounded by a semaphore and results are consumed with
``asyncio.as_completed`` so each case is persisted the moment it is available:

* one structured ``Case`` JSON per document under ``output_dir/cases/``
* a live ``output_dir/progress.csv`` mapping each source ``.txt`` to its
  output ``.json`` alongside success/failure detail, flushed after every case
* a final ``output_dir/manifest.json`` summarising the whole run
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar, override

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from hklii_psla.config import settings
from hklii_psla.experiments.experiment import (
    BaseExperiment,
    ExperimentBatch,
    ExperimentRun,
)
from hklii_psla.extractor.base import ExtractionResult
from hklii_psla.extractor.single_pass import SinglePassExtractor
from hklii_psla.model_factory import create_model

logger = logging.getLogger(__name__)
console = Console()

_PROGRESS_COLUMNS = [
    "source_file",
    "output_json_file",
    "success",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "error",
]


class Exp5Config(BaseModel):
    id: str
    model: str
    provider: str
    case_files_path: str = "test_documents"
    case_files_glob: str = "*.txt"
    output_dir: str
    prompt_style: str = "zero-shot"
    max_concurrency: int | None = None


class CorpusExtractionExperiment(
    BaseExperiment[Exp5Config], experiment_id="corpus-extraction"
):

    _config: Exp5Config
    _config_model: ClassVar[type[BaseModel]] = Exp5Config

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.batch: ExperimentBatch = ExperimentBatch(experiment_id=self._experiment_id)
        self._cases_dir = Path(self._config.output_dir) / "cases"
        self._progress_path = Path(self._config.output_dir) / "progress.csv"
        self._progress_file: Any = None
        self._progress_writer: Any = None

    # -- pipeline ---------------------------------------------------------
    @override
    def run(self) -> None:
        case_files = self._collect_case_files()

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._cases_dir.mkdir(parents=True, exist_ok=True)

        llm = create_model(self._config.model)
        extractor = SinglePassExtractor(llm, model_name=self._config.model)

        max_concurrency = (
            self._config.max_concurrency
            if self._config.max_concurrency is not None
            else settings.MAX_CONCURRENT_REQUESTS
        )

        console.print(
            f"[bold]Extracting {len(case_files)} documents "
            f"(max_concurrency={max_concurrency})...[/bold]"
        )

        self._open_progress()
        try:
            asyncio.run(
                self._extract_corpus(case_files, extractor, max_concurrency)
            )
        finally:
            self._close_progress()

    async def _extract_corpus(
        self,
        case_files: list[Path],
        extractor: SinglePassExtractor,
        max_concurrency: int,
    ) -> None:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _extract_one(file: Path) -> tuple[Path, ExtractionResult]:
            async with semaphore:
                try:
                    judgment_text = extractor.load_judgment(file)
                    result = await extractor.aextract(
                        judgment_text, prompt_style=self._config.prompt_style
                    )
                except Exception as exc:
                    logger.error(f"Extraction failed for {file.name}: {exc}")
                    result = ExtractionResult(
                        case=None,
                        model_name=self._config.model,
                        prompt_style=self._config.prompt_style,
                        error=str(exc),
                    )
                return file, result

        tasks = [asyncio.create_task(_extract_one(f)) for f in case_files]

        for coro in asyncio.as_completed(tasks):
            file, result = await coro
            self._persist_result(file, result)

    def _persist_result(self, file: Path, result: ExtractionResult) -> None:
        """Write one case JSON and one progress row as soon as it is ready."""
        output_name: str | None = None
        if result.success and result.case is not None:
            output_name = self._generate_filename(result, fallback=file.stem)
            case_json = result.case.model_dump(mode="json")
            if result.extraction_metadata is not None:
                case_json["extraction_metadata"] = result.extraction_metadata.to_dict()
            self._cases_dir.mkdir(parents=True, exist_ok=True)
            case_path = self._cases_dir / output_name
            with open(case_path, "w", encoding="utf-8") as f:
                json.dump(case_json, f, indent=2, ensure_ascii=False)

        self._write_progress_row(file, output_name, result)

        self.batch.runs.append(
            ExperimentRun(
                provider=self._config.provider,
                model=self._config.model,
                strategy="single-pass",
                prompt_style=self._config.prompt_style,
                case_file=str(file),
                result=result,
                filename=output_name,
            )
        )

        if result.success:
            logger.info(
                f"    OK — {file.name}: {result.duration_ms:.0f}ms, "
                f"{result.token_count.total_tokens} tokens"
            )
        else:
            logger.info(f"    FAIL — {file.name}: {result.error}")

    # -- progress csv -----------------------------------------------------
    def _open_progress(self) -> None:
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._progress_file = open(
            self._progress_path, "w", encoding="utf-8", newline=""
        )
        self._progress_writer = csv.writer(self._progress_file)
        self._progress_writer.writerow(_PROGRESS_COLUMNS)
        self._progress_file.flush()

    def _write_progress_row(
        self, file: Path, output_name: str | None, result: ExtractionResult
    ) -> None:
        if self._progress_writer is None:
            return
        tc = result.token_count
        self._progress_writer.writerow([
            file.name,
            output_name or "",
            result.success,
            f"{result.duration_ms:.0f}",
            tc.input_tokens,
            tc.output_tokens,
            tc.total_tokens,
            result.error or "",
        ])
        self._progress_file.flush()

    def _close_progress(self) -> None:
        if self._progress_file is not None:
            self._progress_file.close()
            self._progress_file = None
            self._progress_writer = None

    # -- helpers ----------------------------------------------------------
    def _collect_case_files(self) -> list[Path]:
        directory = Path(self._config.case_files_path)
        if not directory.is_dir():
            raise RuntimeError(
                f"case_files_path is not a directory: {self._config.case_files_path}"
            )
        case_files = sorted(directory.glob(self._config.case_files_glob))
        if not case_files:
            raise RuntimeError(
                f"No {self._config.case_files_glob} files found under "
                f"{self._config.case_files_path}"
            )
        return case_files

    def _generate_filename(self, result: ExtractionResult, fallback: str) -> str:
        if (
            result.case
            and result.case.metadata
            and result.case.metadata.neutral_citation
        ):
            safe = re.sub(
                r"[^\w\-]+", "_", result.case.metadata.neutral_citation
            ).strip("_")
            if safe:
                return f"{safe}.json"
        return f"{self._safe_stem(fallback)}.json"

    @staticmethod
    def _safe_stem(value: str) -> str:
        return re.sub(r"[^\w\-]+", "_", value).strip("_")[:120] or "case"

    # -- persistence ------------------------------------------------------
    @override
    def save_results(self) -> None:
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / "manifest.json"
        manifest = {
            "experiment_id": self._experiment_id,
            "summary": self.batch.summary(),
            "runs": [run.to_dict() for run in self.batch.runs],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

        summary = self.batch.summary()
        table = Table(title="Experiment 5 — Async Corpus Extraction")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        for label, value in [
            ("Total documents", summary["total_runs"]),
            ("Successful", summary["successful"]),
            ("Failed", summary["failed"]),
            ("Total duration (ms)", f"{summary['total_duration_ms']:.0f}"),
            ("Total tokens", summary["total_tokens"]),
        ]:
            table.add_row(label, str(value))
        console.print(table)

        console.print(f"\n[bold green]Cases saved to {self._cases_dir}[/bold green]")
        console.print(f"[bold green]Progress CSV: {self._progress_path}[/bold green]")
        console.print(f"[bold green]Manifest: {manifest_path}[/bold green]")