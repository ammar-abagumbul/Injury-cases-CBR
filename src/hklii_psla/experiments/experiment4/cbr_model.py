"""Experiment 4 — flat features to a myCBR model.

Builds a single-concept CBR model (``"PSLACase"``) from the flat
:class:`~hklii_psla.experiments.experiment4.case_loader.CaseFeatures` records:

* ``gender``            — symbol, exact match
* ``age_at_accident``   — integer, polynomial (degree 2) distance
* ``age_at_trial``      — integer, polynomial (degree 2) distance
* ``injuries``          — multiple symbol, ICD-11 taxonomy + symmetric max-overlap
* ``losses``            — multiple symbol, exact match + Jaccard over the sets
* ``psla_amount``       — solution attribute, inactive (never contributes)

``case_id`` is only the instance name.  The query and every candidate share the
one concept, so the project's inheritance taxonomy is a no-op at retrieval time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, override

from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
from hklii_psla.cbr.core.model.double_desc import DoubleDesc
from hklii_psla.cbr.core.model.integer_desc import IntegerDesc
from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
from hklii_psla.cbr.core.project import Project
from hklii_psla.cbr.core.similarity.config.amalgamation_config import AmalgamationConfig
from hklii_psla.cbr.core.similarity.config.multiple_config import (
    MainType,
    MultipleConfig,
    Reuse,
    Type,
)
from hklii_psla.cbr.core.similarity.config.number_config import NumberConfig
from hklii_psla.cbr.core.similarity.similarity import Similarity
from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct
from hklii_psla.experiments.experiment4.case_loader import CaseFeatures
from hklii_psla.experiments.experiment4.icd_similarity import IcdCodeSimFct, IcdTaxonomy
from hklii_psla.schemas import ALL_LOSS_CATEGORIES

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.default_case_base import DefaultCaseBase
    from hklii_psla.cbr.core.model.concept import Concept


class _InnerFct(Protocol):
    """Structural type of the per-element function a ``MultipleConfig`` wraps."""

    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity: ...


CONCEPT_ID = "PSLACase"
CASE_BASE_NAME = "candidates"

GENDER_VALUES = {"Male", "Female"}
AGE_MIN, AGE_MAX = 0, 120
PSLA_MIN, PSLA_MAX = 0.0, 1e12

# Attributes that contribute to similarity (everything except the identifier
# ``case_id`` and the inactive solution attribute ``psla_amount``).
SIMILARITY_ATTRIBUTES = (
    "gender",
    "age_at_accident",
    "age_at_trial",
    "injuries",
    "losses",
)


class MaxOverlapMultipleConfig(MultipleConfig):
    """Symmetric maximum-overlap similarity for two set-valued attributes.

        Sim(A, B) = 1/2 * ( mean_{a in A} max_{b in B} s(a, b)
                          + mean_{b in B} max_{a in A} s(a, b) )

    This mirrors :func:`case_loader.max_overlap_similarity`.  myCBR's built-in
    ``MultipleConfig`` only offers asymmetric BEST_MATCH / PARTNER_* rules, so
    the directed averages are combined here instead.
    """

    def __init__(
        self,
        main_type: MainType = MainType.BEST_MATCH,
        reuse: Reuse = Reuse.REUSE,
        type_: Type = Type.AVG,
    ) -> None:
        super().__init__(main_type, reuse, type_)

    @override
    def calculate_similarity(
        self,
        inner_fct: _InnerFct,
        value1: MultipleAttribute,
        value2: MultipleAttribute,
    ) -> Similarity:
        a = value1.values
        b = value2.values
        if not a and not b:
            return Similarity.get(1.0)
        if not a or not b:
            return Similarity.get(0.0)

        def directed(x: list[Attribute], y: list[Attribute]) -> float:
            return sum(
                max(inner_fct.calculate_similarity(xi, yj).value for yj in y)
                for xi in x
            ) / len(x)

        return Similarity.get(0.5 * (directed(a, b) + directed(b, a)))


class JaccardMultipleConfig(MultipleConfig):
    """Jaccard similarity over the value sets of two multiple attributes."""

    def __init__(
        self,
        main_type: MainType = MainType.BEST_MATCH,
        reuse: Reuse = Reuse.REUSE,
        type_: Type = Type.AVG,
    ) -> None:
        super().__init__(main_type, reuse, type_)

    @override
    def calculate_similarity(
        self,
        inner_fct: _InnerFct,
        value1: MultipleAttribute,
        value2: MultipleAttribute,
    ) -> Similarity:
        a = {v.get_value_as_string() for v in value1.values}
        b = {v.get_value_as_string() for v in value2.values}
        if not a and not b:
            return Similarity.get(1.0)
        if not a or not b:
            return Similarity.get(0.0)
        return Similarity.get(len(a & b) / len(a | b))


def _multiple(desc: SymbolDesc, values: Sequence[str]) -> MultipleAttribute | None:
    """Build a :class:`MultipleAttribute` from raw values, dropping unknowns."""
    atts = [a for a in (desc.get_attribute(v) for v in values) if a is not None]
    if not atts:
        return None
    return MultipleAttribute(desc, atts)


def build_model(
    query: CaseFeatures,
    candidates: Sequence[CaseFeatures],
    weights: dict[str, float] | None = None,
) -> tuple[Project, Concept, DefaultCaseBase, Instance]:
    """Build the project, concept, case base and query instance.

    ``weights`` optionally overrides the per-attribute amalgamation weights;
    any attribute not listed keeps the default weight of ``1.0``.
    """
    prj = Project()
    concept = prj.create_top_concept(CONCEPT_ID)

    injury_codes = sorted({code for c in (query, *candidates) for code in c.injuries})

    gender_desc = SymbolDesc(concept, "gender", GENDER_VALUES)
    age_acc_desc = IntegerDesc(concept, "age_at_accident", AGE_MIN, AGE_MAX)
    age_trial_desc = IntegerDesc(concept, "age_at_trial", AGE_MIN, AGE_MAX)
    injuries_desc = SymbolDesc(concept, "injuries", injury_codes)
    losses_desc = SymbolDesc(concept, "losses", ALL_LOSS_CATEGORIES)
    psla_desc = DoubleDesc(concept, "psla_amount", PSLA_MIN, PSLA_MAX)
    psla_desc.is_solution = True

    # ``is_multiple`` must be set before any instance is created.
    injuries_desc.is_multiple = True
    losses_desc.is_multiple = True

    # -- local similarity functions -------------------------------------
    taxonomy = IcdTaxonomy.from_json()
    icd_fct = IcdCodeSimFct(prj, injuries_desc, taxonomy)
    icd_fct.mc = MaxOverlapMultipleConfig()
    injuries_desc.add_function(icd_fct, active=True)

    losses_fct = losses_desc.get_fct(Project.DEFAULT_FCT_NAME)
    if not isinstance(losses_fct, SymbolFct):
        losses_fct = losses_desc.add_symbol_fct(Project.DEFAULT_FCT_NAME, True)
    losses_fct.mc = JaccardMultipleConfig()

    # -- amalgamation ----------------------------------------------------
    amalgam = concept.get_active_amalgam_fct()
    assert amalgam is not None
    amalgam.set_type(AmalgamationConfig.WEIGHTED_SUM)
    amalgam.set_active(psla_desc, False)

    for desc in (age_acc_desc, age_trial_desc):
        fct = desc.add_integer_fct(f"{desc.name}_sim", False)
        fct.set_function_type_r(NumberConfig.POLYNOMIAL_WITH)
        fct.set_function_parameter_r(2.0)
        fct.set_function_type_l(NumberConfig.POLYNOMIAL_WITH)
        fct.set_function_parameter_l(2.0)
        amalgam.set_active_fct(desc, fct)

    for name in SIMILARITY_ATTRIBUTES:
        amalgam.set_weight(name, (weights or {}).get(name, 1.0))

    # -- instances -------------------------------------------------------
    def populate(inst: Instance, case: CaseFeatures) -> None:
        if case.gender:
            inst.add_attribute(gender_desc, case.gender)
        if case.age_at_accident is not None:
            inst.add_attribute(age_acc_desc, case.age_at_accident)
        if case.age_at_trial is not None:
            inst.add_attribute(age_trial_desc, case.age_at_trial)

        injuries = _multiple(injuries_desc, case.injuries)
        if injuries is not None:
            inst.add_attribute(injuries_desc, injuries)

        losses = _multiple(losses_desc, case.losses)
        if losses is not None:
            inst.add_attribute(losses_desc, losses)

        if case.psla_amount is not None:
            inst.add_attribute(psla_desc, case.psla_amount)

    cb = prj.create_default_cb(CASE_BASE_NAME)
    for case in candidates:
        inst = concept.create_instance(case.case_id)
        populate(inst, case)
        cb.add_case(inst)

    query_inst = concept.get_query_instance()
    populate(query_inst, query)

    return prj, concept, cb, query_inst
