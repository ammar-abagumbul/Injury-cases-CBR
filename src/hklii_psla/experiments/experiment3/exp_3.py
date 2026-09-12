"""Experiment 3 — comparable case extraction.

For each seed judgment: extract it (single-pass), then for every PSLA
comparable case it cites, fetch that comparable case's own judgment from
the Hong Kong Judiciary Legal Reference System and extract it too, so
comparable cases end up as fully structured ``Case`` records rather than
just the few benchmarking fields quoted in the primary judgment.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, override

from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from hklii_psla.experiments.experiment import BaseExperiment, ExperimentBatch, ExperimentRun
from hklii_psla.extractor.base import ExtractionResult
from hklii_psla.extractor.single_pass import SinglePassExtractor
from hklii_psla.judgement_retriever.models import CaseQuery, Judgment
from hklii_psla.judgement_retriever.sources.judiciary import JudiciaryClient
from hklii_psla.judgement_retriever.storage import save_judgment
from hklii_psla.model_factory import create_model
from hklii_psla.schemas import Case, PSLAComparableCase

logger = logging.getLogger(__name__)
console = Console()

_LINK_CSV_COLUMNS = [
    "source_file_name",
    "source_neutral_citation",
    "source_action_number",
    "comparable_file_name",
    "comparable_neutral_citation",
    "comparable_action_number",
    "fetch_status",
    "extraction_status",
    "comparable_case_json_file",
]


class Exp3Config(BaseModel):
    id: str
    model: str
    provider: str
    case_files_path: str
    case_files_format: Literal["raw", "extracted"] = "raw"
    output_dir: str
    max_comparable_cases: int | None = None
    fetch_delay_seconds: float = 1.5
    fetch_timeout: float = 30.0


@dataclass
class ComparableLink:
    """Records what happened when chasing down one comparable case.

    ``comparable_citation`` / ``comparable_action_number`` hold the best
    known identifiers for the comparable: its own extracted metadata when
    available, otherwise the identifiers cited in the primary judgment.
    """

    source_file_name: str
    source_citation: str | None
    source_action_number: str | None
    comparable_case_name: str
    comparable_citation: str | None
    comparable_action_number: str | None
    fetch_status: str  # "fetched" | "not_found" | "error" | "skipped_no_identifier"
    fetch_error: str | None
    judgment_file_name: str | None
    extraction_status: str  # "success" | "failed" | "not_attempted"
    extraction_error: str | None
    case_json_file_name: str | None


class ComparableCaseExtractionExperiment(
    BaseExperiment[Exp3Config], experiment_id="comparable-case-extraction"
):

    _config: Exp3Config
    _config_model: ClassVar[type[BaseModel]] = Exp3Config

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.batch: ExperimentBatch = ExperimentBatch(experiment_id=self._experiment_id)
        self._links: list[ComparableLink] = []

    @override
    def run(self) -> None:
        output_dir = Path(self._config.output_dir)
        primary_dir = output_dir / "primary"
        comparable_judgments_dir = output_dir / "comparable_judgments"
        comparable_cases_dir = output_dir / "comparable_cases"
        for d in (primary_dir, comparable_judgments_dir, comparable_cases_dir):
            d.mkdir(parents=True, exist_ok=True)

        llm = create_model(self._config.model)
        extractor = SinglePassExtractor(llm, model_name=self._config.model)

        case_files = self._collect_case_files()

        fetch_cache: dict[str, Judgment | None] = {}
        extract_cache: dict[str, ExtractionResult] = {}

        with JudiciaryClient(timeout=self._config.fetch_timeout) as judiciary:
            for file in case_files:
                if self._config.case_files_format == "raw":
                    result = self._extract_primary(file, extractor)
                    primary_filename = self._generate_filename(result, fallback=file.stem)
                    self._save_case(result, primary_dir / primary_filename)
                    strategy, prompt_style = "single-pass", "zero-shot"
                else:
                    result = self._load_extracted_case(file)
                    primary_filename = None
                    strategy, prompt_style = "pre-extracted", "n/a"

                self.batch.runs.append(ExperimentRun(
                    provider=self._config.provider,
                    model=self._config.model,
                    strategy=strategy,
                    prompt_style=prompt_style,
                    case_file=str(file),
                    result=result,
                    filename=primary_filename,
                ))

                if result.success:
                    if self._config.case_files_format == "raw":
                        logger.info(f"    OK — primary {file.name}: {result.duration_ms:.0f}ms")
                    else:
                        logger.info(f"    LOADED — pre-extracted primary {file.name}")
                else:
                    logger.info(f"    FAIL — primary {file.name}: {result.error}")
                    continue

                assert result.case is not None
                primary_citation = result.case.metadata.neutral_citation
                primary_action_number = result.case.metadata.action_number or None
                comparable_cases = result.case.psla.comparable_cases
                if self._config.max_comparable_cases is not None:
                    comparable_cases = comparable_cases[: self._config.max_comparable_cases]

                for cc in comparable_cases:
                    self._process_comparable_case(
                        judiciary=judiciary,
                        extractor=extractor,
                        cc=cc,
                        primary_file=file,
                        primary_citation=primary_citation,
                        primary_action_number=primary_action_number,
                        comparable_judgments_dir=comparable_judgments_dir,
                        comparable_cases_dir=comparable_cases_dir,
                        fetch_cache=fetch_cache,
                        extract_cache=extract_cache,
                    )

    def _collect_case_files(self) -> list[Path]:
        """Resolve input files according to ``case_files_format``.

        "raw" expects a directory of judgment ``.txt`` files; "extracted"
        expects a directory of pre-extracted ``Case`` ``.json`` files. A mix
        of both extensions is rejected so half the directory is never
        silently ignored.
        """
        directory = Path(self._config.case_files_path)
        if not directory.is_dir():
            raise RuntimeError(f"case_files_path is not a directory: {self._config.case_files_path}")

        expected, other = (
            ("*.txt", "*.json")
            if self._config.case_files_format == "raw"
            else ("*.json", "*.txt")
        )
        case_files = sorted(directory.glob(expected))
        if not case_files:
            raise RuntimeError(
                f"No {expected} files found under {self._config.case_files_path} "
                f"(case_files_format={self._config.case_files_format!r})"
            )
        if list(directory.glob(other)):
            raise RuntimeError(
                f"case_files_path contains a mix of .txt and .json files; "
                f"case_files_format={self._config.case_files_format!r} requires only {expected} files"
            )
        return case_files

    def _extract_primary(self, file: Path, extractor: SinglePassExtractor) -> ExtractionResult:
        judgment_text = extractor.load_judgment(file)
        return extractor.extract(judgment_text, prompt_style="zero-shot")

    def _load_extracted_case(self, file: Path) -> ExtractionResult:
        """Load a pre-extracted ``Case`` JSON without re-running extraction."""
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            case = Case.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to load pre-extracted case {file.name}: {e}")
            return ExtractionResult(case=None, model_name="pre-extracted", error=str(e))
        return ExtractionResult(case=case, model_name="pre-extracted")

    def _process_comparable_case(
        self,
        *,
        judiciary: JudiciaryClient,
        extractor: SinglePassExtractor,
        cc: PSLAComparableCase,
        primary_file: Path,
        primary_citation: str | None,
        primary_action_number: str | None,
        comparable_judgments_dir: Path,
        comparable_cases_dir: Path,
        fetch_cache: dict[str, Judgment | None],
        extract_cache: dict[str, ExtractionResult],
    ) -> None:
        base = {
            "source_file_name": primary_file.name,
            "source_citation": primary_citation,
            "source_action_number": primary_action_number,
            "comparable_case_name": cc.case_name,
            "comparable_citation": cc.neutral_citation,
            "comparable_action_number": cc.action_number,
        }

        key = self._comparable_key(cc)
        if key is None:
            self._append_link(base, fetch_status="skipped_no_identifier")
            return

        if key in fetch_cache:
            judgment = fetch_cache[key]
        else:
            judgment = self._fetch_comparable(judiciary, cc)
            fetch_cache[key] = judgment
            time.sleep(self._config.fetch_delay_seconds)

        if judgment is None:
            self._append_link(
                base,
                fetch_status="not_found",
                fetch_error="No judgment found for supplied identifiers",
            )
            return

        stem = self._safe_stem(key)
        try:
            judgment_path = save_judgment(judgment, output_dir=comparable_judgments_dir)
        except ValueError as e:
            self._append_link(base, fetch_status="error", fetch_error=str(e))
            return

        if key in extract_cache:
            comp_result = extract_cache[key]
        else:
            comp_result = extractor.extract(judgment.text, prompt_style="zero-shot")
            extract_cache[key] = comp_result

        case_json_path: Path | None = None
        if comp_result.success:
            comp_filename = self._generate_filename(comp_result, fallback=stem)
            case_json_path = comparable_cases_dir / comp_filename
            self._save_case(comp_result, case_json_path)

        # Prefer the comparable's own extracted identifiers, falling back
        # to how it was cited in the primary judgment.
        comp_case = comp_result.case if comp_result.success else None
        comp_citation = (comp_case.metadata.neutral_citation if comp_case else "") or cc.neutral_citation
        comp_action_number = (comp_case.metadata.action_number if comp_case else "") or cc.action_number

        self._append_link(
            base,
            fetch_status="fetched",
            judgment_file_name=judgment_path.name,
            extraction_status="success" if comp_result.success else "failed",
            extraction_error=comp_result.error,
            case_json_file_name=case_json_path.name if case_json_path else None,
            comparable_citation=comp_citation,
            comparable_action_number=comp_action_number,
        )

    def _append_link(self, base: dict[str, Any], **overrides: Any) -> None:
        """Append a ``ComparableLink``, filling unset fields with neutral defaults.

        ``overrides`` win over ``base`` via dict merging, so callers can
        refine base values (e.g. replace cited identifiers with extracted
        ones) without colliding on duplicate keyword arguments.
        """
        self._links.append(ComparableLink(**{
            "fetch_error": None,
            "judgment_file_name": None,
            "extraction_status": "not_attempted",
            "extraction_error": None,
            "case_json_file_name": None,
            **base,
            **overrides,
        }))

    def _fetch_comparable(self, client: JudiciaryClient, cc: PSLAComparableCase) -> Judgment | None:
        try:
            query = CaseQuery(
                case_name=cc.case_name or None,
                neutral_citation=cc.neutral_citation,
                action_number=cc.action_number,
            )
        except ValueError:
            return None
        try:
            return client.search_and_retrieve(query)
        except Exception as e:
            logger.warning(f"Failed to fetch comparable case {cc.case_name!r}: {e}")
            return None

    def _comparable_key(self, cc: PSLAComparableCase) -> str | None:
        if cc.action_number and cc.action_number.strip():
            return f"action:{cc.action_number.strip()}"
        if cc.neutral_citation and cc.neutral_citation.strip():
            return f"citation:{cc.neutral_citation.strip()}"
        if cc.case_name and cc.case_name.strip():
            return f"name:{cc.case_name.strip()}"
        return None

    def _safe_stem(self, key: str) -> str:
        return re.sub(r"[^\w\-]+", "_", key).strip("_")[:120]

    def _save_case(self, result: ExtractionResult, path: Path) -> None:
        if result.case is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.case.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    def _generate_filename(self, result: ExtractionResult, fallback: str) -> str:
        if result.case and result.case.metadata and result.case.metadata.neutral_citation:
            safe = re.sub(r"[^\w\-]+", "_", result.case.metadata.neutral_citation).strip("_")
            if safe:
                return f"{safe}.json"
        return f"{self._safe_stem(fallback)}.json"

    @override
    def save_results(self) -> None:
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([asdict(link) for link in self._links], f, indent=2, ensure_ascii=False)

        links_csv_path = self._write_links_csv(output_dir)

        total_primary = len(self.batch.runs)
        successful_primary = sum(1 for r in self.batch.runs if r.result and r.result.success)
        total_comparable = len(self._links)
        fetched = sum(1 for link in self._links if link.fetch_status == "fetched")
        not_found = sum(1 for link in self._links if link.fetch_status == "not_found")
        fetch_errors = sum(1 for link in self._links if link.fetch_status == "error")
        skipped = sum(1 for link in self._links if link.fetch_status == "skipped_no_identifier")
        extracted_ok = sum(1 for link in self._links if link.extraction_status == "success")

        table = Table(title="Experiment 3 — Comparable Case Extraction")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="yellow")
        for label, value in [
            ("Primary cases processed", total_primary),
            ("Primary extractions succeeded", successful_primary),
            ("Comparable cases cited", total_comparable),
            ("Comparable judgments fetched", fetched),
            ("Comparable judgments not found", not_found),
            ("Comparable judgments fetch errors", fetch_errors),
            ("Comparable cases skipped (no identifier)", skipped),
            ("Comparable cases extracted successfully", extracted_ok),
        ]:
            table.add_row(label, str(value))
        console.print(table)

        console.print(f"\n[bold green]Results saved to {output_dir}[/bold green]")
        console.print(f"[bold green]Manifest: {manifest_path}[/bold green]")
        console.print(f"[bold green]Links CSV: {links_csv_path}[/bold green]")

    def _write_links_csv(self, output_dir: Path) -> Path:
        """Write the source-to-comparable relationship table.

        One row per (source case, cited comparable) pair, including fetch
        and extraction failures, so the table is a complete record. Missing
        values are written as empty cells.
        """
        csv_path = output_dir / "comparable_links.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_LINK_CSV_COLUMNS)
            for link in self._links:
                writer.writerow([
                    link.source_file_name,
                    link.source_citation or "",
                    link.source_action_number or "",
                    link.judgment_file_name or "",
                    link.comparable_citation or "",
                    link.comparable_action_number or "",
                    link.fetch_status,
                    link.extraction_status,
                    link.case_json_file_name or "",
                ])
        return csv_path
