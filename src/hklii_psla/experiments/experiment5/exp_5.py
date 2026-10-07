"""Experiment 5 — asynchronous corpus extraction.

Run the async single-pass extractor over a large corpus of judgment ``.txt``
files.  Concurrency is bounded by a semaphore and results are consumed with
``asyncio.as_completed`` so each case is persisted the moment it is available:

* one structured ``Case`` JSON per document under ``output_dir/cases/``
* a live ``output_dir/progress.csv`` mapping each source ``.txt`` to its
  output ``.json`` alongside outcome detail, flushed after every case
* a final ``output_dir/manifest.json`` summarising the whole run

The run is **resumable**: on start, sources whose progress row has
``status == "ok"`` (and whose JSON still exists) are skipped, and the progress
file is rewritten with the current schema while preserving those rows. Any
other row — failed, partial (e.g. Stage 2 ICD failure), timed out, or missing —
is re-processed. Each case is bounded by a hard timeout so a stalled request
can never hang the whole corpus. An optional ``fallback_provider`` retries
Stage 1 failures that look like provider content-filter rejections.
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
    "status",
    "error_stage",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "error",
    "model",
]


def _is_content_filter_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return "content filter" in lowered or "content_filter" in lowered


class Exp5Config(BaseModel):
    id: str
    model: str
    provider: str
    case_files_path: str = "test_documents"
    case_files_glob: str = "*.txt"
    output_dir: str
    prompt_style: str = "zero-shot"
    max_concurrency: int | None = None
    extraction_timeout: int | None = None
    fallback_provider: str | None = None
    resume: bool = True


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
        all_files = self._collect_case_files()

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._cases_dir.mkdir(parents=True, exist_ok=True)

        existing = self._load_existing_progress() if self._config.resume else {}
        existing = {
            source: self._migrate_row(row) for source, row in existing.items()
        }
        terminal = {
            source for source, row in existing.items() if self._is_terminal(row)
        }
        case_files = [f for f in all_files if f.name not in terminal]
        preserved = [existing[f.name] for f in all_files if f.name in terminal]

        if not case_files:
            console.print(
                "[green]Nothing to do — all cases already processed successfully.[/green]"
            )
            return

        primary = self._make_extractor(self._config.provider)
        fallback = None
        if (
            self._config.fallback_provider
            and self._config.fallback_provider != self._config.provider
        ):
            fallback = self._make_extractor(self._config.fallback_provider)

        timeout = (
            self._config.extraction_timeout
            if self._config.extraction_timeout is not None
            else settings.DEFAULT_TIMEOUT
        )
        max_concurrency = (
            self._config.max_concurrency
            if self._config.max_concurrency is not None
            else settings.MAX_CONCURRENT_REQUESTS
        )

        console.print(
            f"[bold]Extracting {len(case_files)} documents "
            f"(of {len(all_files)}; {len(preserved)} already done) "
            f"(max_concurrency={max_concurrency}, timeout={timeout}s)...[/bold]"
        )

        self._open_progress(preserved)
        try:
            asyncio.run(
                self._extract_corpus(
                    case_files, primary, fallback, max_concurrency, timeout
                )
            )
        finally:
            self._close_progress()

    def _make_extractor(self, provider: str) -> SinglePassExtractor:
        llm = create_model(provider)
        model_name = getattr(llm, "model_name", None) or self._config.model
        return SinglePassExtractor(llm, model_name=str(model_name))

    async def _extract_corpus(
        self,
        case_files: list[Path],
        extractor: SinglePassExtractor,
        fallback: SinglePassExtractor | None,
        max_concurrency: int,
        timeout: int,
    ) -> None:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _extract_one(file: Path) -> tuple[Path, ExtractionResult]:
            async with semaphore:
                try:
                    judgment_text = extractor.load_judgment(file)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Could not load {file.name}: {exc}")
                    return file, ExtractionResult(
                        case=None,
                        model_name=extractor.model_name,
                        prompt_style=self._config.prompt_style,
                        error=f"load failed: {exc}",
                        error_stage="stage1",
                    )

                result = await self._run_extractor(
                    judgment_text, extractor, timeout
                )
                if (
                    result.case is None
                    and fallback is not None
                    and _is_content_filter_error(result.error)
                ):
                    logger.warning(
                        f"{file.name}: primary rejected by content filter; "
                        f"retrying with {fallback.model_name}"
                    )
                    fallback_result = await self._run_extractor(
                        judgment_text, fallback, timeout
                    )
                    if fallback_result.case is not None:
                        return file, fallback_result
                    return file, result
                return file, result

        tasks = [asyncio.create_task(_extract_one(f)) for f in case_files]

        for coro in asyncio.as_completed(tasks):
            file, result = await coro
            self._persist_result(file, result)

    async def _run_extractor(
        self,
        judgment_text: str,
        extractor: SinglePassExtractor,
        timeout: int,
    ) -> ExtractionResult:
        """Run one extractor under a hard timeout, normalising failures."""
        try:
            return await asyncio.wait_for(
                extractor.aextract(
                    judgment_text, prompt_style=self._config.prompt_style
                ),
                timeout=timeout,
            )
        except TimeoutError:
            logger.error(f"Extraction timed out after {timeout}s")
            return ExtractionResult(
                case=None,
                model_name=extractor.model_name,
                prompt_style=self._config.prompt_style,
                duration_ms=timeout * 1000,
                error=f"timeout after {timeout}s",
                error_stage="timeout",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Extraction failed: {exc}")
            return ExtractionResult(
                case=None,
                model_name=extractor.model_name,
                prompt_style=self._config.prompt_style,
                error=str(exc),
                error_stage="stage1",
            )

    def _persist_result(self, file: Path, result: ExtractionResult) -> None:
        """Write one case JSON and one progress row as soon as it is ready."""
        output_name: str | None = None
        if result.case is not None:
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
                model=result.model_name or self._config.model,
                strategy="single-pass",
                prompt_style=self._config.prompt_style,
                case_file=str(file),
                result=result,
                filename=output_name,
            )
        )

        if result.status == "ok":
            logger.info(
                f"    OK — {file.name}: {result.duration_ms:.0f}ms, "
                f"{result.token_count.total_tokens} tokens"
            )
        else:
            logger.info(f"    {result.status.upper()} — {file.name}: {result.error}")

    # -- progress csv -----------------------------------------------------
    def _load_existing_progress(self) -> dict[str, dict[str, Any]]:
        if not self._progress_path.exists():
            return {}
        rows: dict[str, dict[str, Any]] = {}
        with self._progress_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                source = (row.get("source_file") or "").strip()
                if source:
                    rows[source] = row
        return rows

    def _is_terminal(self, row: dict[str, Any]) -> bool:
        if (row.get("status") or "").strip() != "ok":
            return False
        if (row.get("error") or "").strip():
            return False
        output_json = (row.get("output_json_file") or "").strip()
        return bool(output_json) and (self._cases_dir / output_json).exists()

    def _migrate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalise a row (possibly from the older schema) to current columns."""
        success_raw = str(row.get("success", "")).strip().lower() == "true"
        error = (row.get("error") or "").strip()
        status = (row.get("status") or "").strip()
        error_stage = (row.get("error_stage") or "").strip()

        if not status:
            if success_raw and not error:
                status = "ok"
            elif success_raw and error:
                status = "partial"
            else:
                status = "failed"
        if not error_stage:
            if status == "partial":
                error_stage = "stage2"
            elif status == "failed":
                error_stage = "stage1"

        return {
            "source_file": row.get("source_file", ""),
            "output_json_file": row.get("output_json_file", ""),
            "success": str(success_raw),
            "status": status,
            "error_stage": error_stage,
            "duration_ms": row.get("duration_ms", ""),
            "input_tokens": row.get("input_tokens", ""),
            "output_tokens": row.get("output_tokens", ""),
            "total_tokens": row.get("total_tokens", ""),
            "error": error,
            "model": row.get("model") or self._config.model,
        }

    def _open_progress(self, preserved_rows: list[dict[str, Any]]) -> None:
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)

        # Keep a one-off backup of the pre-migration file.
        backup = self._progress_path.with_suffix(".csv.bak")
        if self._progress_path.exists() and not backup.exists():
            backup.write_bytes(self._progress_path.read_bytes())

        self._progress_file = open(  # noqa: SIM115 — kept open for the whole run
            self._progress_path, "w", encoding="utf-8", newline=""
        )
        self._progress_writer = csv.writer(self._progress_file)
        self._progress_writer.writerow(_PROGRESS_COLUMNS)

        for row in preserved_rows:
            migrated = self._migrate_row(row)
            self._progress_writer.writerow(
                [migrated[column] for column in _PROGRESS_COLUMNS]
            )
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
            str(result.case is not None),
            result.status,
            result.error_stage or "",
            f"{result.duration_ms:.0f}",
            tc.input_tokens,
            tc.output_tokens,
            tc.total_tokens,
            result.error or "",
            result.model_name or self._config.model,
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
