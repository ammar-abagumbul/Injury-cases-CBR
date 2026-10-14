"""Experiment 5 — asynchronous corpus extraction.

Run the async single-pass extractor over a large corpus of judgment ``.txt``
files.  Concurrency is bounded by a semaphore and results are consumed with
``asyncio.as_completed`` so each case is persisted the moment it is available:

* one structured ``Case`` JSON per document under ``output_dir/cases/``
* a live ``output_dir/progress.csv`` mapping each source ``.txt`` to its
  output ``.json`` alongside success/failure detail, flushed after every case
* a final ``output_dir/manifest.json`` summarising the latest run

The corpus is append-only. With ``skip_existing`` (the default) a document is
skipped, before any LLM call, when its case is already in ``cases/``; the
check is keyed on the neutral citation derived from the source filename
(``HKDC_2016_1322...`` -> ``[2016] HKDC 1322``) because the citation the LLM
extracts is only known after paying for the call, and is occasionally wrong.
As a safety net, an extraction whose citation matches an existing case is
recorded as a duplicate and never overwrites it. ``progress.csv`` accumulates
across runs. Set ``skip_existing: false`` to force re-extraction (overwrites).
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
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


_SOURCE_NAME_RE = re.compile(r"^(HK[A-Z]+)_(\d{4})_(\d+)")


def _norm_key(value: str) -> str:
    """Lowercase alphanumerics only, so '[2016] HKDC 1322' == '2016_HKDC_1322'."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _candidate_keys(file_name: str) -> set[str]:
    """Keys identifying the case a source file holds, derived from its name.

    ``HKDC_2016_1322.html.txt`` yields the neutral-citation key ``2016hkdc1322``;
    the bare-stem key additionally matches cases saved under a fallback name
    (e.g. action-number style files that have no neutral citation).
    """
    keys = {_norm_key(file_name.split(".")[0])}
    match = _SOURCE_NAME_RE.match(file_name)
    if match:
        court, year, number = match.groups()
        keys.add(_norm_key(f"{year}{court}{number}"))
    keys.discard("")
    return keys


class Exp5Config(BaseModel):
    id: str
    model: str
    provider: str
    case_files_path: str = "test_documents"
    case_files_glob: str = "*.txt"
    output_dir: str
    prompt_style: str = "zero-shot"
    max_concurrency: int | None = None
    skip_existing: bool = True


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
        # normalised case key -> filename of the existing case under cases/
        self._known: dict[str, str] = {}
        self._skipped: list[str] = []
        self._duplicates: list[str] = []

    # -- pipeline ---------------------------------------------------------
    @override
    def run(self) -> None:
        case_files = self._collect_case_files()

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._cases_dir.mkdir(parents=True, exist_ok=True)

        case_files = self._select_pending(case_files)
        if self._skipped:
            shown = ", ".join(self._skipped[:10])
            more = f" (+{len(self._skipped) - 10} more)" if len(self._skipped) > 10 else ""
            console.print(
                f"[yellow]Skipping {len(self._skipped)} already-extracted "
                f"document(s): {shown}{more}[/yellow]"
            )
        if not case_files:
            console.print("[bold]Nothing new to extract.[/bold]")
            return

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
        status: str | None = None
        if result.success and result.case is not None:
            output_name = self._generate_filename(result, fallback=file.stem)
            case_key = _norm_key(
                result.case.metadata.neutral_citation or Path(output_name).stem
            )
            existing = self._known.get(case_key) if self._config.skip_existing else None
            if existing is not None:
                # Same case already in the corpus (e.g. the source filename gave no
                # citation to pre-check): never overwrite, just record it.
                status = "duplicate"
                output_name = existing
                self._duplicates.append(file.name)
                logger.warning(
                    f"Duplicate: {file.name} extracted as "
                    f"{result.case.metadata.neutral_citation!r}, already in corpus as "
                    f"{existing}; existing case left untouched"
                )
            else:
                case_json = result.case.model_dump(mode="json")
                if result.extraction_metadata is not None:
                    case_json["extraction_metadata"] = result.extraction_metadata.to_dict()
                self._write_case_atomically(self._cases_dir / output_name, case_json)
                self._remember(file, output_name, case_key)

        self._write_progress_row(file, output_name, result, status)

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

    # -- append-only corpus -------------------------------------------------
    def _select_pending(self, case_files: list[Path]) -> list[Path]:
        """Drop source files whose case is already in the corpus (or repeated
        within this batch), before any LLM call is made."""
        if not self._config.skip_existing:
            return case_files

        self._known = self._load_known()
        seen = set(self._known)
        pending: list[Path] = []
        for file in case_files:
            keys = _candidate_keys(file.name)
            if keys & seen:
                self._skipped.append(file.name)
                continue
            seen |= keys
            pending.append(file)
        return pending

    def _load_known(self) -> dict[str, str]:
        """Map normalised case key -> existing case filename.

        Built from the case JSONs themselves (filename and neutral citation)
        plus the successful rows of ``progress.csv``. The ledger matters
        because the citation the LLM extracts is occasionally garbled, so the
        source filename is the more reliable identifier.
        """
        known: dict[str, str] = {}
        for path in self._cases_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise TypeError("not a JSON object")
            except (OSError, ValueError, TypeError) as exc:
                logger.warning(f"Ignoring unreadable case file {path.name}: {exc}")
                continue
            known[_norm_key(path.stem)] = path.name
            citation = (data.get("metadata") or {}).get("neutral_citation")
            if citation:
                known[_norm_key(citation)] = path.name

        if self._progress_path.exists():
            existing_files = set(known.values())
            with open(self._progress_path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    output = row.get("output_json_file", "")
                    if row.get("success") in ("True", "duplicate") and output in existing_files:
                        for key in _candidate_keys(row.get("source_file", "")):
                            known[key] = output
        known.pop("", None)
        return known

    def _remember(self, file: Path, output_name: str, case_key: str) -> None:
        self._known[case_key] = output_name
        self._known[_norm_key(Path(output_name).stem)] = output_name
        for key in _candidate_keys(file.name):
            self._known[key] = output_name

    @staticmethod
    def _write_case_atomically(path: Path, case_json: dict[str, Any]) -> None:
        """Write via a temp file so a crash can't leave a truncated case that
        later runs would mistake for an extracted one."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(case_json, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    # -- progress csv -----------------------------------------------------
    def _open_progress(self) -> None:
        path = self._progress_path
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        needs_newline = False
        if not needs_header:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) not in (b"\n", b"\r")
        self._progress_file = open(path, "a", encoding="utf-8", newline="")
        if needs_newline:
            self._progress_file.write("\n")
        self._progress_writer = csv.writer(self._progress_file)
        if needs_header:
            self._progress_writer.writerow(_PROGRESS_COLUMNS)
        self._progress_file.flush()

    def _write_progress_row(
        self,
        file: Path,
        output_name: str | None,
        result: ExtractionResult,
        status: str | None = None,
    ) -> None:
        if self._progress_writer is None:
            return
        tc = result.token_count
        self._progress_writer.writerow([
            file.name,
            output_name or "",
            status or result.success,
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

        corpus_size = len(list(self._cases_dir.glob("*.json")))
        manifest_path = output_dir / "manifest.json"
        manifest = {
            "experiment_id": self._experiment_id,
            "note": "Describes the latest run only; progress.csv accumulates across runs.",
            "summary": self.batch.summary(),
            "corpus": {
                "total_cases": corpus_size,
                "skipped_already_extracted": self._skipped,
                "duplicates_not_overwritten": self._duplicates,
            },
            "runs": [run.to_dict() for run in self.batch.runs],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

        summary = self.batch.summary()
        table = Table(title="Experiment 5 — Async Corpus Extraction")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        for label, value in [
            ("Documents extracted this run", summary["total_runs"]),
            ("Successful", summary["successful"]),
            ("Failed", summary["failed"]),
            ("Skipped (already extracted)", len(self._skipped)),
            ("Duplicates (not overwritten)", len(self._duplicates)),
            ("Cases in corpus now", corpus_size),
            ("Total duration (ms)", f"{summary['total_duration_ms']:.0f}"),
            ("Total tokens", summary["total_tokens"]),
        ]:
            table.add_row(label, str(value))
        console.print(table)

        console.print(f"\n[bold green]Cases saved to {self._cases_dir}[/bold green]")
        console.print(f"[bold green]Progress CSV: {self._progress_path}[/bold green]")
        console.print(f"[bold green]Manifest: {manifest_path}[/bold green]")