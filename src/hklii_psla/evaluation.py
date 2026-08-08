"""
Evaluation metrics for Stage 1 extraction experiments.

This module serves as a bridge between the HKLII PSLA project and the
`extract_bench` structured evaluation framework. It:

1. Generates a JSON Schema from the Pydantic `Case` model.
2. Delegates per-field evaluation to `extract_bench`'s `StructuredEvaluator`.
3. Produces both a rich `EvaluationReport` (from extract_bench) and the
   project's own `ExtractionMetrics` summary for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from hklii_psla.schemas import Case


# ---------------------------------------------------------------------------
# Project-specific metrics dataclasses (backward compatible)
# ---------------------------------------------------------------------------

@dataclass
class FeatureMetrics:
    """Metrics for a single feature / field."""
    field: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class ExtractionMetrics:
    """Aggregate metrics comparing extracted Case against gold Case."""

    # Overall
    exact_matches: dict[str, int]  # field_name -> count of exact matches
    total_fields: int
    field_metrics: list[FeatureMetrics]

    # Statistical
    precision_macro: float
    recall_macro: float
    f1_macro: float

    # Injury-Loss relations
    edge_precision: float
    edge_recall: float
    edge_f1: float

    # Runtime
    duration_ms: float
    token_count: int

    # Validation
    validation_passed: bool
    hallucinations_found: int

    # Rich report from extract_bench (None if framework unavailable)
    report: Optional[Any] = None


# ---------------------------------------------------------------------------
# JSON Schema generation from Pydantic Case model
# ---------------------------------------------------------------------------

def _pydantic_field_to_json_schema(field_info, model_class: type) -> dict:
    """Convert a single Pydantic field to a JSON Schema property."""
    annotation = field_info.annotation

    # Handle Optional[X] (Union[X, None])
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    is_optional = origin is not None and type(None) in args
    if is_optional:
        # Get the non-None type
        inner_types = [a for a in args if a is not type(None)]
        if len(inner_types) == 1:
            annotation = inner_types[0]

    # Build base schema
    schema: dict[str, Any] = {}
    if field_info.description:
        schema["description"] = field_info.description

    # Determine type
    if annotation is str:
        schema["type"] = "string"
        # Use exact match instead of LLM-based semantic comparison
        schema["evaluation_config"] = "string_exact"
    elif annotation is int:
        schema["type"] = "integer"
    elif annotation is float:
        schema["type"] = "number"
    elif annotation is bool:
        schema["type"] = "boolean"
    elif hasattr(annotation, "model_fields"):
        # Nested Pydantic model
        schema.update(_pydantic_model_to_json_schema_properties(annotation))
    elif origin is list:
        # list[X] — use skip preset to avoid LLM-based array evaluation
        schema["type"] = "array"
        item_type = args[0] if args else str
        if hasattr(item_type, "model_fields"):
            schema["items"] = _pydantic_model_to_json_schema_properties(item_type)
        elif item_type is str:
            schema["items"] = {"type": "string"}
        elif item_type is int:
            schema["items"] = {"type": "integer"}
        elif item_type is float:
            schema["items"] = {"type": "number"}
        else:
            schema["items"] = {"type": "string"}
        # Skip LLM-based array evaluation; arrays are compared via legacy set logic
        schema["evaluation_config"] = "skip"
    elif hasattr(annotation, "__members__"):
        # Enum
        schema["type"] = "string"
        schema["enum"] = [e.value for e in annotation]
        schema["evaluation_config"] = "string_exact"
    else:
        # Fallback
        schema["type"] = "string"

    # Handle Optional via anyOf
    if is_optional and "type" in schema:
        schema = {"anyOf": [{"type": "null"}, schema]}

    return schema


def _pydantic_model_to_json_schema_properties(model_class: type) -> dict:
    """Convert a Pydantic model to JSON Schema object properties."""
    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, field_info in model_class.model_fields.items():
        field_schema = _pydantic_field_to_json_schema(field_info, model_class)
        properties[name] = field_schema

        # Check if required (not Optional and no default)
        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)
        is_optional = origin is not None and type(None) in getattr(annotation, "__args__", ())
        has_default = field_info.default is not None or field_info.default_factory is not None

        if not is_optional and not has_default:
            required.append(name)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = required

    return result


def generate_case_json_schema() -> dict:
    """Generate a JSON Schema for the Case Pydantic model.

    This schema is used by extract_bench's StructuredEvaluator to
    traverse and evaluate each field.
    """
    properties = _pydantic_model_to_json_schema_properties(Case)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Case",
        "description": "Complete structured representation of a personal injury case.",
        **properties,
    }


# ---------------------------------------------------------------------------
# Bridge to extract_bench
# ---------------------------------------------------------------------------

def _get_extract_bench():
    """Lazy-import extract_bench. Returns None if not available."""
    try:
        import extract_bench  # noqa: F401
        return extract_bench
    except ImportError:
        # extract_bench lives at the project root; add it to sys.path
        import sys
        from pathlib import Path

        # Find the project root (parent of src/)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "extract_bench" / "__init__.py").exists():
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                try:
                    import extract_bench  # noqa: F401
                    return extract_bench
                except ImportError:
                    pass
                break
        return None


def _case_to_dict(case: Case) -> dict:
    """Convert a Case model to a plain dict suitable for extract_bench."""
    return case.model_dump(mode="json", exclude_none=False)


def _build_field_metrics_from_report(report) -> list[FeatureMetrics]:
    """Convert extract_bench FieldOutcome list to FeatureMetrics list."""
    field_metrics: list[FeatureMetrics] = []
    seen_fields: dict[str, dict[str, Any]] = {}

    # Group by normalized path
    for fo in report.field_outcomes:
        key = fo.normalized_path
        if key not in seen_fields:
            seen_fields[key] = {"precision": [], "recall": [], "f1": [], "tp": 0, "fp": 0, "fn": 0}  # type: ignore[assignment]

        # Each field outcome has a single score; for confusion we use pass/fail
        tp_val: int = seen_fields[key]["tp"] + (1 if fo.passed else 0)  # type: ignore[operator]
        fp_val: int = seen_fields[key]["fp"] + (0 if fo.passed else 1)  # type: ignore[operator]
        fn_val: int = seen_fields[key]["fn"] + (0 if fo.passed else 1)  # type: ignore[operator]
        seen_fields[key]["tp"] = tp_val
        seen_fields[key]["fp"] = fp_val
        seen_fields[key]["fn"] = fn_val

        # Score as proxy for precision/recall/f1 at field level
        seen_fields[key]["precision"].append(fo.score)
        seen_fields[key]["recall"].append(fo.score)
        seen_fields[key]["f1"].append(fo.score)

    for feature_field, data in seen_fields.items():
        precision = float(np.mean(data["precision"])) if data["precision"] else 0.0
        recall = float(np.mean(data["recall"])) if data["recall"] else 0.0
        f1 = float(np.mean(data["f1"])) if data["f1"] else 0.0
        field_metrics.append(FeatureMetrics(
            field=feature_field,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=int(data["tp"]),  # type: ignore[arg-type]
            false_positives=int(data["fp"]),  # type: ignore[arg-type]
            false_negatives=int(data["fn"]),  # type: ignore[arg-type]
        ))

    return field_metrics


def _compute_edge_metrics_from_cases(predicted: Case, gold: Case) -> tuple[float, float, float]:
    """Compute precision/recall/F1 for injury-loss relations."""
    pred_edges = {
        (rel.injury.lower().strip(), rel.loss.lower().strip())
        for rel in (predicted.injury_loss_relations or [])
    }
    gold_edges = {
        (rel.injury.lower().strip(), rel.loss.lower().strip())
        for rel in (gold.injury_loss_relations or [])
    }

    tp = len(pred_edges & gold_edges)
    fp = len(pred_edges - gold_edges)
    fn = len(gold_edges - pred_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def _count_array_hallucinations(predicted: Case, gold: Case) -> int:
    """Count spurious items in array fields (items in pred but not in gold)."""
    count = 0

    # Injury-loss relations
    gold_edges = {
        (rel.injury.lower().strip(), rel.loss.lower().strip())
        for rel in (gold.injury_loss_relations or [])
    }
    pred_edges = {
        (rel.injury.lower().strip(), rel.loss.lower().strip())
        for rel in (predicted.injury_loss_relations or [])
    }
    count += len(pred_edges - gold_edges)

    # Injuries
    gold_injuries = {
        (inj.description.lower().strip(), inj.injury_type.value.lower().strip(),
         (inj.body_part or "").lower().strip())
        for inj in (gold.injuries.injuries or [])
    }
    pred_injuries = {
        (inj.description.lower().strip(), inj.injury_type.value.lower().strip(),
         (inj.body_part or "").lower().strip())
        for inj in (predicted.injuries.injuries or [])
    }
    count += len(pred_injuries - gold_injuries)

    # Losses
    gold_losses = {
        (loss.category.lower().strip(), (loss.description or "").lower().strip())
        for loss in (gold.losses.losses or [])
    }
    pred_losses = {
        (loss.category.lower().strip(), (loss.description or "").lower().strip())
        for loss in (predicted.losses.losses or [])
    }
    count += len(pred_losses - gold_losses)

    # Comparable cases
    gold_cases = {
        (c.case_name.lower().strip(), c.neutral_citation.lower().strip())
        for c in (gold.psla.comparable_cases or [])
    }
    pred_cases = {
        (c.case_name.lower().strip(), c.neutral_citation.lower().strip())
        for c in (predicted.psla.comparable_cases or [])
    }
    count += len(pred_cases - gold_cases)

    # Treatment lists
    gold_treatments = set(t.lower().strip() for t in (gold.treatment.treatments_received or []))
    pred_treatments = set(t.lower().strip() for t in (predicted.treatment.treatments_received or []))
    count += len(pred_treatments - gold_treatments)

    gold_future = set(t.lower().strip() for t in (gold.treatment.treatments_future or []))
    pred_future = set(t.lower().strip() for t in (predicted.treatment.treatments_future or []))
    count += len(pred_future - gold_future)

    return count


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    predicted: Optional[Case],
    gold: Case,
    duration_ms: float = 0,
    token_count: int = 0,
    use_extract_bench: bool = True,
) -> ExtractionMetrics:
    """
    Compare predicted Case against gold-standard Case.

    When `use_extract_bench=True` and the extract_bench framework is available,
    evaluation is delegated to extract_bench's StructuredEvaluator for rich
    per-field metrics. Falls back to the legacy manual comparison otherwise.

    Args:
        predicted: The extracted Case (or None if extraction failed).
        gold: The gold-standard benchmark Case.
        duration_ms: Extraction duration in milliseconds.
        token_count: Token count for the extraction.
        use_extract_bench: Whether to attempt using extract_bench framework.

    Returns:
        ExtractionMetrics with per-field and aggregate scores.
    """
    # Handle null prediction
    if predicted is None:
        gold_dict = _case_to_dict(gold)
        n_fields = _count_leaf_fields(gold_dict)
        return ExtractionMetrics(
            exact_matches={},
            total_fields=n_fields,
            field_metrics=[],
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            edge_precision=0.0,
            edge_recall=0.0,
            edge_f1=0.0,
            duration_ms=duration_ms,
            token_count=token_count,
            validation_passed=False,
            hallucinations_found=0,
        )

    # Try extract_bench bridge
    report = None
    if use_extract_bench:
        eb = _get_extract_bench()
        if eb is not None:
            try:
                report = _evaluate_with_extract_bench(predicted, gold)
            except Exception:
                # Fall back to legacy on any error
                pass

    if report is not None:
        # Build metrics from extract_bench report
        field_metrics = _build_field_metrics_from_report(report)

        # Compute macro averages from field outcomes
        scores = [fo.score for fo in report.field_outcomes if fo.score is not None]
        if scores:
            precision_macro = float(np.mean(scores))
            recall_macro = float(np.mean(scores))
            f1_macro = float(np.mean(scores))
        else:
            precision_macro = recall_macro = f1_macro = 0.0

        # Edge metrics still computed manually (injury-loss relations)
        edge_precision, edge_recall, edge_f1 = _compute_edge_metrics_from_cases(predicted, gold)

        # Hallucinations = false positives from scalar fields (report) + array fields (legacy)
        scalar_hallucinations = report.outcomes.confusion.false_positive
        array_hallucinations = _count_array_hallucinations(predicted, gold)
        hallucinations = scalar_hallucinations + array_hallucinations

        return ExtractionMetrics(
            exact_matches={},  # extract_bench doesn't use this legacy concept
            total_fields=report.outcomes.total_evaluated,
            field_metrics=field_metrics,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            edge_precision=edge_precision,
            edge_recall=edge_recall,
            edge_f1=edge_f1,
            duration_ms=duration_ms,
            token_count=token_count,
            validation_passed=report.outcomes.pass_rate > 0,
            hallucinations_found=hallucinations,
            report=report,
        )
    else:
        # Legacy fallback
        return _evaluate_legacy(predicted, gold, duration_ms, token_count)


def _evaluate_with_extract_bench(predicted: Case, gold: Case):
    """Run evaluation through extract_bench and return the EvaluationReport.

    Uses non-LLM metrics for all fields to avoid external API calls.
    Array fields are evaluated with the legacy set-based comparison
    (handled separately via _compute_edge_metrics_from_cases).
    """
    import asyncio

    from extract_bench import (
        AsyncEvaluationConfig,
        StructuredEvaluator,
        StructuredEvaluatorConfig,
    )

    json_schema = generate_case_json_schema()
    gold_dict = _case_to_dict(gold)
    pred_dict = _case_to_dict(predicted)

    # Build evaluator with non-LLM metrics and short timeout
    async_config = AsyncEvaluationConfig(
        metric_timeout_seconds=10,
        n_max_retries=1,
        parallel_traversal=True,
    )
    evaluator = StructuredEvaluator(
        StructuredEvaluatorConfig(metrics=[], async_config=async_config)
    )

    # Run evaluation
    eval_result = asyncio.run(
        evaluator.evaluate_async(json_schema, gold_dict, pred_dict)
    )

    # Build report manually (skip ReportBuilder since it creates its own evaluator)
    from extract_bench.evaluation.reporting.content_stats import (
        collect_content_stats,
        compute_coverage,
    )
    from extract_bench.evaluation.reporting.models import EvaluationReport
    from extract_bench.evaluation.reporting.outcome_stats import collect_outcome_stats
    from extract_bench.evaluation.reporting.schema_stats import collect_schema_stats
    from extract_bench.evaluation.reporting.report_builder import _compute_hash, _generate_output_name

    schema_tree = eval_result["schema"]
    results = eval_result["results"]

    schema_stats = collect_schema_stats(schema_tree)
    gold_stats = collect_content_stats(gold_dict, "gold")
    extracted_stats = collect_content_stats(pred_dict, "extracted")
    coverage = compute_coverage(gold_dict, pred_dict, schema_tree)

    outcome_stats, field_outcomes = collect_outcome_stats(
        results,
        schema_tree,
        max_reasoning_length=200,
        top_n_lowest=5,
    )

    output_name = _generate_output_name(json_schema, gold_dict, pred_dict)

    scores = [f.score for f in field_outcomes if f.score is not None]
    field_score = sum(scores) / len(scores) if scores else 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for f in field_outcomes:
        if f.score is None:
            continue
        if f.array_breakdown:
            weight = max(1, f.array_breakdown.matched + f.array_breakdown.missed_gold)
        else:
            weight = 1
        weighted_sum += f.score * weight
        total_weight += weight
    overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    report = EvaluationReport(
        output_name=output_name,
        timestamp=asyncio.run(_get_timestamp()),
        schema_hash=_compute_hash(json_schema),
        gold_hash=_compute_hash(gold_dict),
        extracted_hash=_compute_hash(pred_dict),
        schema_stats=schema_stats,
        gold_stats=gold_stats,
        extracted_stats=extracted_stats,
        coverage=coverage,
        outcomes=outcome_stats,
        field_outcomes=field_outcomes,
        overall_score=overall_score,
        field_score=field_score,
        overall_pass_rate=outcome_stats.pass_rate,
    )

    return report


async def _get_timestamp() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Legacy evaluation (fallback when extract_bench is unavailable)
# ---------------------------------------------------------------------------

def _compare_scalar(gold: Any, pred: Any) -> bool:
    """Compare two scalar values, handling None and empty defaults."""
    if gold is None and pred in (None, 0, "", 0.0, []):
        return True
    if pred is None and gold in (None, 0, "", 0.0, []):
        return True
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(gold - pred) < 0.01
    return gold == pred


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict with dot-notation keys."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            items[key] = v
        else:
            items[key] = v
    return items


def _count_leaf_fields(d: dict) -> int:
    """Count leaf-level fields in a nested dict."""
    count = 0
    for v in d.values():
        if isinstance(v, dict):
            count += _count_leaf_fields(v)
        else:
            count += 1
    return count


def _is_list_field(field_name: str) -> bool:
    """Check if a field is a list (injuries, losses, comparable_cases, relations)."""
    list_suffixes = ("injuries", "losses", "comparable_cases", "injury_loss_relations")
    return field_name.endswith(list_suffixes)


def _make_hashable(val):
    """Recursively convert values to hashable equivalents."""
    if isinstance(val, list):
        return tuple(_make_hashable(v) for v in val)
    if isinstance(val, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
    return val


def _to_comparable_set(val) -> set:
    """Convert a list value to a set of hashable items for comparison."""
    if val is None:
        return set()
    if not isinstance(val, list):
        return {val} if val else set()

    result = set()
    for item in val:
        if isinstance(item, dict):
            hashable = tuple(sorted((k, _make_hashable(v)) for k, v in item.items()))
            result.add(hashable)
        else:
            result.add(item)
    return result


def _evaluate_legacy(
    predicted: Case,
    gold: Case,
    duration_ms: float = 0,
    token_count: int = 0,
) -> ExtractionMetrics:
    """Legacy manual evaluation (used as fallback)."""
    pred_dict = predicted.model_dump()
    gold_dict = gold.model_dump()

    pred_flat = _flatten_dict(pred_dict)
    gold_flat = _flatten_dict(gold_dict)

    all_fields = sorted(set(list(pred_flat.keys()) + list(gold_flat.keys())))

    field_metrics: list[FeatureMetrics] = []
    exact_matches: dict[str, int] = {}
    hallucinations = 0

    for f_name in all_fields:
        g_val = gold_flat.get(f_name)
        p_val = pred_flat.get(f_name)

        if _is_list_field(f_name):
            g_set = _to_comparable_set(g_val)
            p_set = _to_comparable_set(p_val)

            tp = len(g_set & p_set)
            fp = len(p_set - g_set)
            fn = len(g_set - p_set)

            hallucinations += fp

            if tp + fp + fn == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            field_metrics.append(FeatureMetrics(
                field=f_name, precision=precision, recall=recall, f1=f1,
                true_positives=tp, false_positives=fp, false_negatives=fn,
            ))
        else:
            matched = _compare_scalar(g_val, p_val)
            tp = 1 if matched else 0
            fp = 0 if matched else 1
            fn = 0 if matched else 1
            exact_matches[f_name] = tp

            field_metrics.append(FeatureMetrics(
                field=f_name,
                precision=1.0 if matched else 0.0,
                recall=1.0 if matched else 0.0,
                f1=1.0 if matched else 0.0,
                true_positives=tp, false_positives=fp, false_negatives=fn,
            ))

    if field_metrics:
        precision_macro = float(np.mean([m.precision for m in field_metrics]))
        recall_macro = float(np.mean([m.recall for m in field_metrics]))
        f1_macro = float(np.mean([m.f1 for m in field_metrics]))
    else:
        precision_macro = recall_macro = f1_macro = 0.0

    edge_precision, edge_recall, edge_f1 = _compute_edge_metrics_from_cases(predicted, gold)

    return ExtractionMetrics(
        exact_matches=exact_matches,
        total_fields=len(all_fields),
        field_metrics=field_metrics,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        edge_precision=edge_precision,
        edge_recall=edge_recall,
        edge_f1=edge_f1,
        duration_ms=duration_ms,
        token_count=token_count,
        validation_passed=True,
        hallucinations_found=hallucinations,
    )