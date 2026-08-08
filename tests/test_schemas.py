"""Tests for Pydantic schemas."""

import json
from datetime import date

import pytest

from hklii_psla.schemas import (
    Case,
    CaseMetadata,
    PlaintiffBackground,
    Injury,
    InjurySummary,
    InjuryType,
    InjuryCategory,
    Treatment,
    Loss,
    LossSummary,
    PSLAAward,
    PSLAComparableCase,
    PSLAComparison,
    DeathInfo,
    InjuryLossRelation,
    ALL_LOSS_CATEGORIES,
)


class TestCaseMetadata:
    def test_defaults(self):
        m = CaseMetadata()
        assert m.neutral_citation == ""
        assert m.plaintiff_count == 1
        assert m.has_pre_existing_injuries is False

    def test_full_init(self):
        m = CaseMetadata(
            neutral_citation="[2020] HKDC 1745",
            action_number="DCPI 2723/2018",
            case_name="LIU WEIGUANG v. LI KENG KO AND ANOTHER",
            judgment_date=date(2020, 6, 15),
            plaintiff_count=1,
            has_pre_existing_injuries=False,
        )
        assert m.neutral_citation == "[2020] HKDC 1745"
        assert m.judgment_date == date(2020, 6, 15)

    def test_json_serialization(self):
        m = CaseMetadata(
            neutral_citation="[2020] HKDC 1745",
            judgment_date=date(2020, 6, 15),
        )
        d = m.model_dump(mode="json")
        assert d["judgment_date"] == "2020-06-15"


class TestPlaintiffBackground:
    def test_defaults(self):
        p = PlaintiffBackground()
        assert p.gender is None
        assert p.age_at_accident is None

    def test_with_data(self):
        p = PlaintiffBackground(
            gender="Male",
            age_at_accident=45,
            occupation_before="Driver",
            salary_before=20000.0,
        )
        assert p.gender == "Male"
        assert p.age_at_accident == 45
        assert p.salary_before == 20000.0


class TestInjury:
    def test_valid(self):
        i = Injury(
            description="Fractured left tibia",
            injury_type=InjuryType.TEMPORARY,
            body_part="left leg",
        )
        assert i.injury_type == InjuryType.TEMPORARY

    def test_injury_summary(self):
        s = InjurySummary(
            injuries=[
                Injury(description="Fracture", injury_type=InjuryType.TEMPORARY),
            ],
            overall_category=InjuryCategory.SERIOUS,
        )
        assert len(s.injuries) == 1
        assert s.overall_category == InjuryCategory.SERIOUS


class TestTreatment:
    def test_defaults(self):
        t = Treatment()
        assert t.treatments_received == []
        assert t.hospitalisation_days is None

    def test_with_data(self):
        t = Treatment(
            treatments_received=["physiotherapy", "surgery"],
            hospitalisation_days=21,
            operations_count=2,
            sick_leave_days_actual=180,
        )
        assert len(t.treatments_received) == 2
        assert t.hospitalisation_days == 21
        assert t.operations_count == 2


class TestLoss:
    def test_valid(self):
        l = Loss(
            category="Loss of mobility",
            description="Unable to walk long distances",
            present=True,
        )
        assert l.category == "Loss of mobility"
        assert l.present is True


class TestPSLA:
    def test_default(self):
        p = PSLAAward()
        assert p.amount is None
        assert p.comparable_cases == []

    def test_with_comparables(self):
        p = PSLAAward(
            amount=500000.0,
            comparable_cases=[
                PSLAComparableCase(
                    case_name="Test v Test",
                    neutral_citation="[2019] HKDC 100",
                    action_number="DCPI 100/2019",
                    injury_description="Fractured leg",
                    hospitalisation_days=30,
                    operations_count=1,
                    comparison=PSLAComparison.SIMILAR,
                    psla_amount=450000.0,
                )
            ],
        )
        assert p.amount == 500000.0
        assert len(p.comparable_cases) == 1
        assert p.comparable_cases[0].comparison == PSLAComparison.SIMILAR


class TestDeathInfo:
    def test_default(self):
        d = DeathInfo()
        assert d.conscious_before_death is False

    def test_with_data(self):
        d = DeathInfo(
            conscious_before_death=True,
            hours_between_accident_and_death=6,
            days_between_accident_and_death=0,
        )
        assert d.conscious_before_death is True
        assert d.hours_between_accident_and_death == 6


class TestInjuryLossRelation:
    def test_valid(self):
        r = InjuryLossRelation(
            injury="Rotator cuff tear",
            loss="Loss of congenial employment",
            evidence=["paragraph 62"],
        )
        assert r.injury == "Rotator cuff tear"
        assert len(r.evidence) == 1


class TestCase:
    def test_full_case(self):
        c = Case(
            metadata=CaseMetadata(
                neutral_citation="[2020] HKDC 1745",
                action_number="DCPI 2723/2018",
                case_name="TEST v TEST",
            ),
            plaintiff=PlaintiffBackground(gender="Male", age_at_accident=45),
            injuries=InjurySummary(
                injuries=[
                    Injury(description="Fracture", injury_type=InjuryType.TEMPORARY)
                ]
            ),
            treatment=Treatment(hospitalisation_days=14),
            psla=PSLAAward(amount=500000.0),
            injury_loss_relations=[
                InjuryLossRelation(
                    injury="Fracture",
                    loss="Loss of mobility",
                    evidence=["para 10"],
                )
            ],
        )
        d = c.model_dump(mode="json")
        assert d["metadata"]["neutral_citation"] == "[2020] HKDC 1745"
        assert d["plaintiff"]["gender"] == "Male"
        assert len(d["injury_loss_relations"]) == 1

    def test_json_roundtrip(self):
        c = Case(
            metadata=CaseMetadata(case_name="Test Case"),
            plaintiff=PlaintiffBackground(gender="Male"),
        )
        json_str = json.dumps(c.model_dump(mode="json"))
        d = json.loads(json_str)
        reconstructed = Case.model_validate(d)
        assert reconstructed.metadata.case_name == "Test Case"
        assert reconstructed.plaintiff.gender == "Male"


class TestLossCategories:
    def test_count(self):
        assert len(ALL_LOSS_CATEGORIES) == 27

    def test_contains_key_items(self):
        assert "Loss of mobility" in ALL_LOSS_CATEGORIES
        assert "Loss of congenial employment" in ALL_LOSS_CATEGORIES
        assert "Loss of the senses" in ALL_LOSS_CATEGORIES
        assert "Personality change" in ALL_LOSS_CATEGORIES
        assert "Loss of family life" in ALL_LOSS_CATEGORIES
