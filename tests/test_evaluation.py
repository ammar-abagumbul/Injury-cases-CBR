"""Tests for evaluation module."""

import pytest

from hklii_psla.schemas import (
    Case,
    CaseMetadata,
    PlaintiffBackground,
    Injury,
    InjurySummary,
    InjuryType,
    Treatment,
    PSLAAward,
    InjuryLossRelation,
)
from hklii_psla.evaluation import evaluate


def make_minimal_case(**kwargs):
    """Helper to create a minimal Case for testing."""
    return Case(
        metadata=CaseMetadata(case_name=kwargs.get("case_name", "Test")),
        plaintiff=PlaintiffBackground(gender=kwargs.get("gender", "Male")),
        injuries=InjurySummary(
            injuries=kwargs.get("injuries", [])
        ),
        treatment=Treatment(),
        psla=PSLAAward(),
        injury_loss_relations=kwargs.get("relations", []),
    )


class TestEvaluate:
    def test_perfect_match(self):
        gold = make_minimal_case(
            case_name="Test Case",
            gender="Male",
        )
        pred = make_minimal_case(
            case_name="Test Case",
            gender="Male",
        )
        metrics = evaluate(pred, gold)
        assert metrics.f1_macro == 1.0
        assert metrics.precision_macro == 1.0
        assert metrics.recall_macro == 1.0

    def test_null_prediction(self):
        gold = make_minimal_case(case_name="Test")
        metrics = evaluate(None, gold)
        assert metrics.f1_macro == 0.0
        assert metrics.validation_passed is False

    def test_edge_metrics_perfect(self):
        gold = make_minimal_case(
            relations=[
                InjuryLossRelation(
                    injury="Fracture",
                    loss="Loss of mobility",
                    evidence=["para 10"],
                )
            ]
        )
        pred = make_minimal_case(
            relations=[
                InjuryLossRelation(
                    injury="Fracture",
                    loss="Loss of mobility",
                    evidence=["para 10"],
                )
            ]
        )
        metrics = evaluate(pred, gold)
        assert metrics.edge_precision == 1.0
        assert metrics.edge_recall == 1.0
        assert metrics.edge_f1 == 1.0

    def test_edge_metrics_missing(self):
        gold = make_minimal_case(
            relations=[
                InjuryLossRelation(
                    injury="Fracture",
                    loss="Loss of mobility",
                )
            ]
        )
        pred = make_minimal_case()  # no relations
        metrics = evaluate(pred, gold)
        assert metrics.edge_precision == 0.0
        assert metrics.edge_recall == 0.0
        assert metrics.edge_f1 == 0.0

    def test_hallucination_count(self):
        gold = make_minimal_case()
        pred = make_minimal_case(
            relations=[
                InjuryLossRelation(
                    injury="Made-up injury",
                    loss="Loss of mobility",
                )
            ]
        )
        metrics = evaluate(pred, gold)
        assert metrics.hallucinations_found == 1
