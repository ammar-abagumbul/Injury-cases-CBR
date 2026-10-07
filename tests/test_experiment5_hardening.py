"""Tests for Experiment 5 resumable/timeout hardening."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from hklii_psla.experiments.experiment5.exp_5 import (
    _PROGRESS_COLUMNS,
    CorpusExtractionExperiment,
    _is_content_filter_error,
)
from hklii_psla.extractor.base import ExtractionResult
from hklii_psla.schemas import Case


def _make_experiment(
    tmp_path: Path,
    *,
    source_names: list[str] | None = None,
    progress_rows: list[dict[str, str]] | None = None,
    config_extra: dict | None = None,
) -> tuple[CorpusExtractionExperiment, Path]:
    src = tmp_path / "src"
    src.mkdir()
    for name in source_names or []:
        (src / name).write_text("judgment", encoding="utf-8")

    out = tmp_path / "out"
    config = {
        "id": "corpus-extraction",
        "model": "gpt",
        "provider": "gpt",
        "case_files_path": str(src),
        "output_dir": str(out),
    }
    config.update(config_extra or {})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    experiment = CorpusExtractionExperiment(config_path)
    experiment._cases_dir.mkdir(parents=True, exist_ok=True)

    if progress_rows is not None:
        experiment._progress_path.parent.mkdir(parents=True, exist_ok=True)
        with experiment._progress_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(progress_rows[0].keys()))
            writer.writeheader()
            writer.writerows(progress_rows)
    return experiment, out


def test_is_content_filter_error():
    assert _is_content_filter_error("... rejected by the content filter")
    assert _is_content_filter_error("Error: content_filter triggered")
    assert not _is_content_filter_error("connection reset")
    assert not _is_content_filter_error(None)


def test_extraction_result_status():
    case = Case()
    assert ExtractionResult(case=case).status == "ok"
    assert ExtractionResult(case=case, error="stage2 boom").status == "partial"
    assert ExtractionResult(case=None, error="boom").status == "failed"
    assert ExtractionResult(case=None, error="late", error_stage="timeout").status == "timeout"


def test_migrate_old_rows(tmp_path):
    experiment, _ = _make_experiment(tmp_path)
    assert experiment._migrate_row({"success": "True", "error": ""})["status"] == "ok"
    partial = experiment._migrate_row(
        {"success": "True", "error": "Stage 2 ... failed"}
    )
    assert partial["status"] == "partial"
    assert partial["error_stage"] == "stage2"
    failed = experiment._migrate_row({"success": "False", "error": "content filter"})
    assert failed["status"] == "failed"
    assert failed["error_stage"] == "stage1"


def test_load_existing_progress_dedupes_by_source(tmp_path):
    rows = [
        {"source_file": "a.txt", "output_json_file": "", "success": "False", "error": "x"},
        {"source_file": "a.txt", "output_json_file": "a.json", "success": "True", "error": ""},
        {"source_file": "b.txt", "output_json_file": "", "success": "False", "error": "y"},
    ]
    experiment, _ = _make_experiment(tmp_path, progress_rows=rows)
    loaded = experiment._load_existing_progress()
    assert set(loaded) == {"a.txt", "b.txt"}
    assert loaded["a.txt"]["success"] == "True"


def test_is_terminal(tmp_path):
    experiment, _ = _make_experiment(
        tmp_path,
        source_names=["a.txt"],
        progress_rows=[
            {
                "source_file": "a.txt",
                "output_json_file": "a.json",
                "success": "True",
                "error": "",
            }
        ],
    )
    (experiment._cases_dir / "a.json").write_text("{}", encoding="utf-8")
    row = experiment._load_existing_progress()["a.txt"]
    assert experiment._is_terminal(experiment._migrate_row(row))

    # a partial (stage-2) row is not terminal and must be re-processed
    partial = {
        "source_file": "a.txt",
        "output_json_file": "a.json",
        "success": "True",
        "error": "Stage 2 failed",
    }
    assert not experiment._is_terminal(experiment._migrate_row(partial))


def test_open_progress_preserves_rows_and_upgrades_schema(tmp_path):
    old_rows = [
        {"source_file": "a.txt", "output_json_file": "a.json", "success": "True", "error": ""},
        {"source_file": "b.txt", "output_json_file": "", "success": "False", "error": "boom"},
    ]
    experiment, _ = _make_experiment(tmp_path, progress_rows=old_rows)
    experiment._open_progress([old_rows[0]])
    experiment._close_progress()

    with experiment._progress_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == _PROGRESS_COLUMNS
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["source_file"] == "a.txt"
    assert rows[0]["status"] == "ok"

    # the pre-migration file is backed up once
    backup = experiment._progress_path.with_suffix(".csv.bak")
    assert backup.exists()


def test_write_progress_row_uses_status_and_model(tmp_path):
    experiment, _ = _make_experiment(tmp_path)
    experiment._open_progress([])
    result = ExtractionResult(case=Case(), model_name="test-model")
    experiment._write_progress_row(Path("a.txt"), "a.json", result)
    experiment._close_progress()

    with experiment._progress_path.open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["status"] == "ok"
    assert row["error_stage"] == ""
    assert row["model"] == "test-model"
    assert row["success"] == "True"
