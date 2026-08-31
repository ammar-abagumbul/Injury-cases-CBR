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

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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


class PSLAComparison(str, Enum):
    MORE_SERIOUS = "more serious"
    SIMILAR = "similar or same"
    LESS_SERIOUS = "less serious"


class LossCategory(str, Enum):
    BODILY_INTEGRITY = "Loss of bodily integrity"
    SENSES = "Loss of the senses"
    SCARRING_DISFIGUREMENT = "Loss from scarring or disfigurement"
    MOBILITY = "Loss of mobility"
    INDEPENDENCE = "Loss of independence"
    DOMESTIC_TASKS = "Loss of ability to do domestic tasks"
    MENTAL_ABILITY = "Loss of mental ability"
    COMMUNICATION_ABILITY = "Loss of communication ability"
    PERSONALITY_CHANGE = "Personality change"
    PUBLIC_CONFIDENCE = "Loss of confidence in going out in the public"
    FOOD_DRINK_ENJOYMENT = "Loss of ability to enjoy food and drink"
    QUIET_SOLITUDE = "Loss of ability to enjoy quiet or solitude"
    FAMILY_LIFE = "Loss of family life"
    PARENTHOOD_GRANDPARENTHOOD = (
        "Loss of ability to participate in parenthood or grandparenthood"
    )
    MARRIAGE_PROSPECTS = "Loss of marriage prospects"
    FAMILY_BREAKDOWN = "Breakdown of the family"
    SEXUAL_FUNCTION = "Loss of sexual function and sexual life"
    GIVE_BIRTH = "Loss of ability to give birth"
    NATURAL_CHILDBIRTH = "Loss of ability to give natural childbirth"
    SOCIAL_LIFE = "Loss of social life"
    SPORTS_HOBBIES = "Loss of ability to engage in sports and hobbies"
    HOLIDAYS_SPECIAL_OCCASIONS = "Loss of holidays or special occasions"
    CONGENIAL_EMPLOYMENT = "Loss of congenial employment"
    EDUCATION = "Loss of ability to pursue education"
    VOLUNTEER_COMMUNITY = (
        "Loss of ability to volunteer or engage in community service"
    )
    EXPECTATION_OF_LIFE = "Loss of expectation of life"


class CaseMetadata(BaseModel):
    """Top-level case identification."""

    neutral_citation: str = Field(
        default="",
        description="Neutral citation, e.g. '[2020] HKDC 1745'",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )
    action_number: str = Field(
        default="",
        description="Action number, e.g. 'DCPI 2723/2018'",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )
    case_name: str = Field(
        default="",
        description="Case name, e.g. 'LIU WEIGUANG v. LI KENG KO AND ANOTHER'",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_fuzzy"}]} }
    )
    judgment_date: str | None= Field(
        default=None,
        description="Date of judgment in the format YYYY-MM-DD",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )
    plaintiff_count: int = Field(
        default=1,
        description="Number of plaintiffs",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    has_pre_existing_injuries: bool = Field(
        default=False,
        description="Whether the case involves pre-existing injuries",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "boolean_exact"}]} }
    )


class PlaintiffBackground(BaseModel):
    """Demographic and occupational profile of the plaintiff."""

    gender: Gender | None = Field(
        default=None,
        description="Gender of the plaintiff",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )
    age_at_accident: int | None = Field(
        default=None, description="Age at time of accident",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    age_at_trial: int | None = Field(
        default=None, description="Age at time of trial / assessment",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    occupation_before: str | None = Field(
        default=None, description="Occupation before accident",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )
    occupation_after: str | None = Field(
        default=None, description="Occupation after accident",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )
    expected_occupation_after: str | None = Field(
        default=None, description="Expected occupation after accident",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )
    salary_before: float | None = Field(
        default=None, description="Monthly salary before accident (HKD)",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    salary_after: float | None = Field(
        default=None, description="Monthly salary after accident (HKD)",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    expected_salary_after: float | None = Field(
        default=None, description="Expected monthly salary after accident (HKD)",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    education_level: str | None = Field(
        default=None, description="Education level of plaintiff",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )


class ClinicalManifestation(BaseModel):
    """
    A clinical symptom, sign, complaint, functional consequence, or neurological
    deficit resulting from an injury.

    Examples:
    - dysphasia (consequence of brain injury)
    - radiculopathy (consequence of disc herniation)
    - reduced range of motion (consequence of fracture)
    - severe headaches, dizziness, or tinnitus
    """
    description: str = Field(
        description="The symptom, sign, or functional deficit (e.g., 'Dysphasia', 'Chronic lower back pain')",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )
    manifestation_type: Literal["symptom", "sign", "functional_deficit", "neurological_deficit"] = Field(
        description="Categorization of the manifestation",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )
    caused_by_injury_id: str | None = Field(
        default=None,
        description="The unique ID of the specific Injury that caused this manifestation, if identifiable from the text.",
        json_schema_extra={ "evaluation_config": "skip" }
    )


class Injury(BaseModel):
    """
    A specific diagnosed or documented physical injury suffered by the plaintiff.

    Extract only the underlying structural injury or pathological condition itself.
    Do NOT extract symptoms, signs, complaints, or functional consequences here;
    those belong in ClinicalManifestation.

    Examples of injuries:
    - fracture of the distal fibula
    - torn ACL
    - cervical disc herniation
    - concussion / traumatic brain injury
    - acute traumatic subdural hemorrhage

    Do NOT extract here:
    - pain, tenderness, swelling, bruising, stiffness, numbness, reduced ROM, dysphasia

    Order the injuries by their occurrence in the body following the order:
        1) head
        2) neck
        3) thorax
        4) abdomen, lower back, lumbar spine or pelvis
        5) the shoulder or upper arm
        6) the elbow or forearm
        7) the wrist or hand
        8) the hip or thigh
        9) the knee or lower leg
        10) the ankle or foot

    """
    injury_id: str = Field(
        description=(
            "Generate a unique short string identifier generated during extraction "
            "following the format: 'inj_001', 'inj_002', etc... to map relationships."
        ),
        json_schema_extra={ "evaluation_config": "skip" }
    )
    description: str = Field(description="Description of the injury")
    injury_type: InjuryType = Field(description="Type of the injury")
    body_part: str | None = Field(
        default=None, description="Body part affected, e.g. 'left ankle'",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )
    laterality: Literal["left", "right", "bilateral"] | None = Field(
        default=None, description="Laterality of the injury",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )
    source: list[str] = Field(
        default_factory=list,
        description=(
            "Source of the injury from the case. This should be a verbatim "
            "description of the injury as it appears in the case report, "
            "not just the paragraph number. The text should be an exact match "
            "to what appears in the case report. DO NOT shorten or summarize the text. "
            "DO NOT use three dots to indicate ellipsis."
        )
    )
    icd_code: str | None = Field(
        default=None,
        description="ICD-11 code assigned during Stage 2 classification",
        json_schema_extra={ "evaluation_config": "skip" }
    )
    icd_description: str | None = Field(
        default=None,
        description="ICD-11 description assigned during Stage 2 classification",
        json_schema_extra={ "evaluation_config": "skip" }
    )


class InjurySummary(BaseModel):
    """Collected injury information and their related manifestations for the case."""

    injuries: list[Injury] = Field(
        default_factory=list,
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "array_llm"}]} }
    )
    manifestations: list[ClinicalManifestation] = Field(
        default_factory=list,
        description="List of symptoms and clinical consequences.",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "array_llm"}]} }
    )
    overall_category: InjuryCategory | None = Field(
        default=None,
        description="Overall injury severity category",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )


class Treatment(BaseModel):
    """Treatment received or planned."""

    treatments_received: list[str] = Field(
        default_factory=list,
        description="Types of treatment received, e.g. 'physiotherapy'",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "array_llm"}]} }
    )
    treatments_future: list[str] = Field(
        default_factory=list,
        description="Types of treatment to receive in future",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "array_llm"}]} }
    )
    hospitalisation_days: int | None = Field(
        default=None, description="Days of hospitalisation received",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    expected_hospitalisation_days: int | None = Field(
        default=None, description="Expected days of hospitalisation in future",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    operations_count: int | None = Field(
        default=None, description="Number of operations received",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    future_operations_count: int | None = Field(
        default=None, description="Number of operations expected in future",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    sick_leave_days_actual: int | None = Field(
        default=None, description="Actual days of sick leave",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    sick_leave_days_expected: int | None = Field(
        default=None, description="Expected days of sick leave",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )


class Loss(BaseModel):
    """
    A single loss of amenity identified in the judgment.

    A loss of amenity describes a way in which the plaintiff's ability to
    enjoy life, relationships, activities, independence, or personal
    fulfilment has been reduced because of the injury.

    CATEGORY DEFINITIONS
    --------------------

    Loss of bodily integrity:
        Loss, impairment, mutilation, or reduced function of a body part,
        organ, eye, limb, or bodily function as a whole.

    Loss of the senses:
        Loss or reduction of sight, hearing, smell, taste, or touch.

    Loss from scarring or disfigurement:
        Reduced enjoyment of life caused by visible scarring, burns,
        deformity, or disfigurement, including embarrassment,
        self-consciousness, reduced confidence, or avoidance of activities.

    Loss of mobility:
        Reduced ability to move independently, including walking,
        climbing stairs, cycling, travelling, or moving around freely.

    Loss of independence:
        Reduced ability to live or care for oneself independently, including
        needing assistance with washing, dressing, feeding, personal care,
        or moving safely.

    Loss of ability to do domestic tasks:
        Reduced ability to perform ordinary household activities such as
        cooking, cleaning, shopping, childcare, gardening, or home
        maintenance.

    Loss of mental ability:
        Reduced ability to use memory, concentration, reasoning,
        understanding, intellectual ability, or mental capacity.

    Loss of communication ability:
        Reduced ability to communicate or express oneself through speaking,
        writing, reading, understanding language, or interacting effectively
        with others.

    Personality change:
        Changes in mood, behaviour, temperament, or character that reduce
        enjoyment of life or the ability to maintain relationships.

    Loss of confidence in going out in the public:
        Reduced ability or willingness to leave home or be in public because
        of fear, anxiety, embarrassment, or loss of confidence.

    Loss of ability to enjoy food and drink:
        Reduced ability to enjoy meals because of impaired taste, digestion,
        or inability to consume certain foods.

    Loss of ability to enjoy quiet or solitude:
        Reduced ability to enjoy peace or quiet because of increased
        sensitivity to noise or psychological distress.

    Loss of family life:
        Reduced ability to enjoy ordinary family life, including family
        activities, outings, routines, conversation, affection, or interaction.

    Loss of ability to participate in parenthood or grandparenthood:
        Reduced ability to enjoy or participate in parental or grandparental
        activities, including caring for, playing with, supporting, or
        sharing experiences with children or grandchildren.

    Loss of marriage prospects:
        Reduced prospects of marrying or forming a long-term relationship
        because of the injury, disability, disfigurement, or its consequences.

    Breakdown of the family:
        Breakdown of a marriage, partnership, cohabiting relationship, or
        family unit caused or materially contributed to by the injury.

    Loss of sexual function and sexual life:
        Reduced ability to enjoy or participate in sexual relations or
        activities, including effects such as pain, impotence, loss of
        libido, or psychological inhibition.

    Loss of ability to give birth:
        Loss of the ability to have children because of infertility, loss of
        reproductive capacity, or another injury-related consequence.

    Loss of ability to give natural childbirth:
        Loss of the opportunity or experience of vaginal/natural childbirth
        because injury-related consequences require intervention such as
        caesarean delivery or otherwise prevent natural childbirth.

    Loss of social life:
        Reduced ability to enjoy ordinary social interaction and
        participation, including visiting friends, attending events,
        restaurants, clubs, community activities, or similar activities.

    Loss of ability to engage in sports and hobbies:
        Reduced ability to participate in recreational activities previously
        enjoyed, including sports, hobbies, dancing, crafts, music, gardening,
        driving, reading, or playing an instrument.

    Loss of holidays or special occasions:
        Loss, interruption, or reduced enjoyment of holidays, trips,
        celebrations, social events, or other anticipated occasions.

    Loss of congenial employment:
        Loss of the ability to continue or pursue work that was enjoyable,
        meaningful, rewarding, or important to the plaintiff's sense of
        fulfilment or professional identity.

    Loss of ability to pursue education:
        Reduced ability to continue studies or acquire new skills.

    Loss of ability to volunteer or engage in community service:
        Reduced ability to contribute to society through charitable,
        volunteer, or community activities.

    Loss of expectation of life:
        Shortening of the plaintiff's expected lifespan as a result of the
        injury.

    The categories are based on the project's Categories of Loss of
    Amenities taxonomy.
    """

    category: LossCategory = Field(
        description="Loss category name",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )
    description: str = Field(
        description="Specific loss or limitation supported by the judgment",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )
    caused_by_injury_id: str = Field(
        description="The unique ID of the specific Injury that caused this loss, if identifiable from the text.",
    )

class LossSummary(BaseModel):
    """All losses identified in the case."""

    losses: list[Loss] = Field(
        default_factory=list,
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )


class PSLAComparableCase(BaseModel):
    """A comparable case cited for PSLA benchmarking. """

    case_name: str = Field(
        default="", description="Case name",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_fuzzy"}]} }
    )
    neutral_citation: str | None = Field(
        default=None, description="Neutral citation",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )
    action_number: str | None = Field(
        default=None, description="Action number",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_exact"}]} }
    )
    injury_description: str | None= Field(
        default=None, description="Description of injury",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "string_semantic"}]} }
    )
    hospitalisation_days: int | None = Field(
        default=None,
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    operations_count: int | None = Field(
        default=None,
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    sick_leave_days: int | None = Field(
        default=None,
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    comparison: PSLAComparison | None = Field(
        default=None,
        description="Injury severity compared to present case",
    )
    psla_amount: float | None = Field(
        default=None, description="PSLA award amount (HKD)",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )


class PSLAAward(BaseModel):
    """PSLA award details."""

    amount: float | None = Field(default=None, description="PSLA award amount (HKD)")
    comparable_cases: list[PSLAComparableCase] = Field(
        default_factory=list,
        description="There should be at least one comparable case. Only leave empty if you are sure no comparable cases are available.",
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "array_llm"}]} }
    )


class DeathInfo(BaseModel):
    """Information about victim's death (only if applicable)."""

    conscious_before_death: bool = Field(
        default=False,
        description="Victim was alive/conscious for some time before death",
    )
    hours_between_accident_and_death: int | None = Field(
        default=None,
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )
    days_between_accident_and_death: int | None = Field(
        default=None,
        json_schema_extra={ "evaluation_config": {"metrics": [{"metric_id": "number_exact"}]} }
    )


class InjuryLossRelation(BaseModel):
    """Explicitly supported relation between an injury and a loss."""

    injury: str = Field(description="Injury description")
    caused_by_injury_id: str | None = Field(
        default=None,
        description="The unique ID of the specific Injury that caused this loss, if identifiable from the text.",
    )
    loss: LossCategory = Field(
        description="Loss category",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes from judgment supporting the relationship",
    )


class Case(BaseModel):
    """Complete structured representation of a personal injury case."""

    metadata: CaseMetadata = Field(default_factory=CaseMetadata)
    plaintiff: PlaintiffBackground = Field(default_factory=PlaintiffBackground)
    injuries: InjurySummary = Field(default_factory=InjurySummary)
    treatment: Treatment = Field(default_factory=Treatment)
    losses: LossSummary = Field(default_factory=LossSummary)
    psla: PSLAAward = Field(default_factory=PSLAAward)
    death: DeathInfo | None = Field(
        default=None,
        json_schema_extra={ "evaluation_config": "skip" }
    )


class InjuryLossRelationList(BaseModel):
    """Wrapper for a list of injury-loss relations.

    Used as the structured output schema for the relation extraction agent
    in section-by-section and multi-agent strategies, since
    `with_structured_output()` requires a single top-level model.
    """

    injury_loss_relations: list[InjuryLossRelation] = Field(
        default_factory=list,
    )


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
]

if __name__ == "__main__":
    import json
    from pathlib import Path

    case_schema = Case.model_json_schema()
    output_file = Path(__file__).parent / "test_schema.json"

    _ = output_file.write_text(json.dumps(case_schema, indent=2))

    print(f"Schema written to {output_file}")
