"""
Pydantic schemas for personal injury case structured extraction.

Covers:
- Case metadata (citation, action number, dates)
- Victim background
- Injuries
- Treatment
- Losses (per REVISED_LOSS.md taxonomy)
- PSLA awards and comparable cases
- InjuryLossRelation
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class InjuryCategory(str, Enum):
    NON_SERIOUS = "Non-serious injury"
    SERIOUS = "Serious injury"
    SUBSTANTIAL = "Substantial injury"
    GROSS_DISABILITY = "Gross disability"
    DISASTER = "Disaster"


class InjuryType(str, Enum):
    PERMANENT = "Permanent injury"
    TEMPORARY = "Temporary injury"
    RESIDUAL = "Residual disability"
    OTHER = "Other injury"


class PSLAComparison(str, Enum):
    MORE_SERIOUS = "more serious"
    SIMILAR = "similar or same"
    LESS_SERIOUS = "less serious"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class CaseMetadata(BaseModel):
    """Top-level case identification."""

    neutral_citation: str = Field(
        default="",
        description="Neutral citation, e.g. '[2020] HKDC 1745'",
    )
    action_number: str = Field(
        default="",
        description="Action number, e.g. 'DCPI 2723/2018'",
    )
    case_name: str = Field(
        default="",
        description="Case name, e.g. 'LIU WEIGUANG v. LI KENG KO AND ANOTHER'",
    )
    judgment_date: Optional[date] = Field(
        default=None,
        description="Date of judgment",
    )
    plaintiff_count: int = Field(
        default=1,
        description="Number of plaintiffs",
    )
    has_pre_existing_injuries: bool = Field(
        default=False,
        description="Whether the case involves pre-existing injuries",
    )


# ---------------------------------------------------------------------------
# Victim / Plaintiff Background
# ---------------------------------------------------------------------------

class PlaintiffBackground(BaseModel):
    """Demographic and occupational profile of the plaintiff."""

    gender: Optional[Gender] = Field(default=None, description="Gender of the plaintiff")
    age_at_accident: Optional[int] = Field(
        default=None, description="Age at time of accident"
    )
    age_at_trial: Optional[int] = Field(
        default=None, description="Age at time of trial / assessment"
    )
    occupation_before: Optional[str] = Field(
        default=None, description="Occupation before accident"
    )
    occupation_after: Optional[str] = Field(
        default=None, description="Occupation after accident"
    )
    expected_occupation_after: Optional[str] = Field(
        default=None, description="Expected occupation after accident"
    )
    salary_before: Optional[float] = Field(
        default=None, description="Monthly salary before accident (HKD)"
    )
    salary_after: Optional[float] = Field(
        default=None, description="Monthly salary after accident (HKD)"
    )
    expected_salary_after: Optional[float] = Field(
        default=None, description="Expected monthly salary after accident (HKD)"
    )
    education_level: Optional[str] = Field(
        default=None, description="Education level of plaintiff"
    )


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

class Injury(BaseModel):
    """
    A specific diagnosed or documented physical injury suffered by the plaintiff.

    Extract only the underlying injury or pathological condition itself.
    Do NOT extract symptoms, signs, complaints, or functional consequences
    as separate injuries.

    Examples of injuries:
    - fracture of the distal fibula
    - torn ACL
    - cervical disc herniation
    - concussion
    - laceration of the scalp

    Do NOT extract:
    - pain
    - tenderness
    - swelling
    - bruising
    - stiffness
    - numbness
    - reduced range of motion
    - headaches
    """

    description: str = Field(description="Description of the injury")
    injury_type: InjuryType = Field(description="Type of the injury")
    body_part: Optional[str] = Field(
        default=None, description="Body part affected, e.g. 'left ankle'"
    )
    laterality: Optional[Literal["left", "right", "bilateral"]] = Field(
        default=None, description="Laterality of the injury"
    )


class InjurySummary(BaseModel):
    """Collected injury information for the case."""

    injuries: list[Injury] = Field(default_factory=list)
    overall_category: Optional[InjuryCategory] = Field(
        default=None, description="Overall injury severity category"
    )


# ---------------------------------------------------------------------------
# Treatment
# ---------------------------------------------------------------------------

class Treatment(BaseModel):
    """Treatment received or planned."""

    treatments_received: list[str] = Field(
        default_factory=list,
        description="Types of treatment received, e.g. 'physiotherapy'",
    )
    treatments_future: list[str] = Field(
        default_factory=list,
        description="Types of treatment to receive in future",
    )
    hospitalisation_days: Optional[int] = Field(
        default=None, description="Days of hospitalisation received"
    )
    expected_hospitalisation_days: Optional[int] = Field(
        default=None, description="Expected days of hospitalisation in future"
    )
    operations_count: Optional[int] = Field(
        default=None, description="Number of operations received"
    )
    future_operations_count: Optional[int] = Field(
        default=None, description="Number of operations expected in future"
    )
    sick_leave_days_actual: Optional[int] = Field(
        default=None, description="Actual days of sick leave"
    )
    sick_leave_days_expected: Optional[int] = Field(
        default=None, description="Expected days of sick leave"
    )


# ---------------------------------------------------------------------------
# Losses  (REVISED_LOSS.md taxonomy)
# ---------------------------------------------------------------------------

class Loss(BaseModel):
    """A single loss category identified in the judgment."""

    category: str = Field(description="Loss category name per REVISED_LOSS.md taxonomy")
    description: Optional[str] = Field(
        default=None, description="Specific details or notes about this loss"
    )
    present: bool = Field(
        default=True, description="Whether this loss was found to be present"
    )


class LossSummary(BaseModel):
    """All losses identified in the case."""

    losses: list[Loss] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PSLA
# ---------------------------------------------------------------------------

class PSLAComparableCase(BaseModel):
    """A comparable case cited for PSLA benchmarking."""

    case_name: str = Field(default="", description="Case name")
    neutral_citation: str = Field(default="", description="Neutral citation")
    action_number: str = Field(default="", description="Action number")
    injury_description: str = Field(default="", description="Description of injury")
    hospitalisation_days: Optional[int] = Field(default=None)
    operations_count: Optional[int] = Field(default=None)
    sick_leave_days: Optional[int] = Field(default=None)
    comparison: Optional[PSLAComparison] = Field(
        default=None,
        description="Injury severity compared to present case",
    )
    psla_amount: Optional[float] = Field(
        default=None, description="PSLA award amount (HKD)"
    )


class PSLAAward(BaseModel):
    """PSLA award details."""

    amount: Optional[float] = Field(default=None, description="PSLA award amount (HKD)")
    comparable_cases: list[PSLAComparableCase] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Death
# ---------------------------------------------------------------------------

class DeathInfo(BaseModel):
    """Information about victim's death (only if applicable)."""

    conscious_before_death: bool = Field(
        default=False,
        description="Victim was alive/conscious for some time before death",
    )
    hours_between_accident_and_death: Optional[int] = Field(default=None)
    days_between_accident_and_death: Optional[int] = Field(default=None)


# ---------------------------------------------------------------------------
# Injury-Loss Relation
# ---------------------------------------------------------------------------

class InjuryLossRelation(BaseModel):
    """Explicitly supported relation between an injury and a loss."""

    injury: str = Field(description="Injury description")
    loss: str = Field(description="Loss category from REVISED_LOSS.md")
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes from judgment supporting the relationship",
    )


# ---------------------------------------------------------------------------
# Top-level Case
# ---------------------------------------------------------------------------

class Case(BaseModel):
    """Complete structured representation of a personal injury case."""

    metadata: CaseMetadata = Field(default_factory=CaseMetadata)
    plaintiff: PlaintiffBackground = Field(default_factory=PlaintiffBackground)
    injuries: InjurySummary = Field(default_factory=InjurySummary)
    treatment: Treatment = Field(default_factory=Treatment)
    losses: LossSummary = Field(default_factory=LossSummary)
    psla: PSLAAward = Field(default_factory=PSLAAward)
    death: Optional[DeathInfo] = Field(default=None)
    injury_loss_relations: list[InjuryLossRelation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Wrapper models for structured output (section-by-section / multi-agent)
# ---------------------------------------------------------------------------

class InjuryLossRelationList(BaseModel):
    """Wrapper for a list of injury-loss relations.

    Used as the structured output schema for the relation extraction agent
    in section-by-section and multi-agent strategies, since
    `with_structured_output()` requires a single top-level model.
    """

    injury_loss_relations: list[InjuryLossRelation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Loss taxonomy (canonical list from REVISED_LOSS.md)
# ---------------------------------------------------------------------------

ALL_LOSS_CATEGORIES: list[str] = [
    "Loss of bodily integrity",
    "Loss of the senses",
    "Loss from scarring or disfigurement",
    "Loss of mobility",
    "Loss of independence",
    "Loss of ability to do domestic tasks",
    "Loss of mental ability",
    "Loss of communication ability",
    "Personality change",
    "Loss of confidence in going out in the public",
    "Loss of ability to enjoy food and drink",
    "Loss of ability to enjoy quiet or solitude",
    "Loss of family life",
    "Loss of ability to participate in parenthood or grandparenthood",
    "Loss of marriage prospects",
    "Breakdown of the family",
    "Loss of sexual function and sexual life",
    "Loss of ability to give birth",
    "Loss of ability to give natural childbirth",
    "Loss of social life",
    "Loss of ability to engage in sports and hobbies",
    "Loss of holidays or special occasions",
    "Loss of congenial employment",
    "Loss of ability to pursue education",
    "Loss of ability to volunteer or engage in community service",
    "Loss of expectation of life",
    "Other",
]
