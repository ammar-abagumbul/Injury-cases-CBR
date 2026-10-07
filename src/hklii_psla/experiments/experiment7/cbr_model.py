"""Experiment 7 — corpus features to a myCBR model.

Builds a single-concept CBR model (``"PSLACase"``) over the whole extracted
corpus.  The local similarity functions are the myCBR ones (identical to
Experiment 4 for the shared attributes):

* ``injuries``           — multiple symbol, ICD-11 taxonomy + symmetric max-overlap
* ``losses``             — multiple symbol, exact match + Jaccard over the sets
* ``age_at_accident``    — integer, polynomial (degree 2) distance
* ``age_at_trial``       — integer, polynomial (degree 2) distance
* ``operations_count``   — integer, polynomial (degree 2) distance
* ``hospitalisation_days`` — integer, polynomial (degree 2) distance
* ``n_injuries``         — integer, polynomial (degree 2) distance
* ``n_losses``           — integer, polynomial (degree 2) distance
* ``gender``             — symbol, exact match (optional; off by default)
* ``overall_category``   — ordered symbol, ordinal severity distance (optional;
  off by default because it is the award band and would encode the target)

``psla_amount`` is a *solution* attribute: stored but deactivated, so it never
contributes to similarity.

**Scoring.**  The global similarity is a clinical-first composite: the clinical
group (``injuries`` + ``losses``) is a missing-*penalised* weighted mean and
carries ``clinical_alpha`` (default 0.7); the auxiliary group is an
available-case weighted mean and adjusts within the clinical neighbourhood.  A
loose injury-similarity gate can drop clinically unrelated candidates first.

**Missing values.**  The corpus is sparse.  myCBR's default special-value
similarity scores ``unknown`` vs ``known`` as 0 and ``unknown`` vs ``unknown``
as 1, which biases retrieval towards other incomplete cases.  Set
``missing_policy="special"`` to recover the raw myCBR amalgamation used by
:meth:`CorpusCBRModel.retrieve_cbr`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
from hklii_psla.cbr.core.model.double_desc import DoubleDesc
from hklii_psla.cbr.core.model.integer_desc import IntegerDesc
from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
from hklii_psla.cbr.core.project import Project
from hklii_psla.cbr.core.retrieval.sequential_retrieval import SequentialRetrieval
from hklii_psla.cbr.core.similarity.config.amalgamation_config import AmalgamationConfig
from hklii_psla.cbr.core.similarity.config.number_config import NumberConfig
from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct
from hklii_psla.experiments.experiment4.cbr_model import (
    JaccardMultipleConfig,
    MaxOverlapMultipleConfig,
)
from hklii_psla.experiments.experiment4.icd_similarity import (
    IcdCodeSimFct,
    IcdTaxonomy,
)
from hklii_psla.experiments.experiment7.features import CorpusCase
from hklii_psla.schemas import ALL_LOSS_CATEGORIES

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.default_case_base import DefaultCaseBase
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.concept import Concept


CONCEPT_ID = "PSLACase"
CASE_BASE_NAME = "corpus"

GENDER_VALUES = {"Male", "Female"}
AGE_MIN, AGE_MAX = 0, 120
PSLA_MIN, PSLA_MAX = 0.0, 1e12
OPS_MAX = 50
HOSP_MAX = 1000
N_INJ_MAX = 50
N_LOSS_MAX = 40

SEVERITY_VALUES: tuple[str, ...] = (
    "Non-serious injury",
    "Serious injury",
    "Substantial injury",
    "Gross disability",
    "Disaster",
)

CORE_ATTRIBUTES: tuple[str, ...] = (
    "gender",
    "age_at_accident",
    "age_at_trial",
    "injuries",
    "losses",
)
EXTRA_ATTRIBUTES: tuple[str, ...] = (
    "overall_category",
    "operations_count",
    "hospitalisation_days",
    "n_injuries",
    "n_losses",
)

# The full attribute universe the model knows how to build.
ALL_ATTRIBUTES: tuple[str, ...] = CORE_ATTRIBUTES + EXTRA_ATTRIBUTES

# The clinical attributes are the backbone of the score: what the plaintiff's
# body and life actually lost.  They are aggregated with a missing-value
# *penalty*, so a case that cannot be compared clinically is pushed down rather
# than let off.
CLINICAL_ATTRIBUTES: tuple[str, ...] = ("injuries", "losses")

# Auxiliary attributes only adjust within a clinical neighbourhood.  They are
# aggregated available-case (missing attributes dropped).
AUXILIARY_ATTRIBUTES: tuple[str, ...] = (
    "age_at_accident",
    "age_at_trial",
    "operations_count",
    "hospitalisation_days",
    "n_injuries",
    "n_losses",
)

# Clinical-first default.  ``gender`` is a near-constant that only compresses the
# ranking; ``overall_category`` is the award band (a target-derived encoding of
# the compensation) and is deliberately kept off the clinical score — it is
# reconstructed from the award for diagnostics and the optional band filter.
DEFAULT_ATTRIBUTES: tuple[str, ...] = CLINICAL_ATTRIBUTES + AUXILIARY_ATTRIBUTES

# Backwards-compatible alias.
SIMILARITY_ATTRIBUTES = CORE_ATTRIBUTES

MissingPolicy = Literal["available", "special"]


def _multiple(desc: SymbolDesc, values: Sequence[str]) -> MultipleAttribute | None:
    """Build a :class:`MultipleAttribute` from raw values, dropping unknowns."""
    atts = [a for a in (desc.get_attribute(v) for v in values) if a is not None]
    if not atts:
        return None
    return MultipleAttribute(desc, atts)


def available_case_similarities(
    S: np.ndarray, known: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Available-case weighted-sum similarity.

    ``S`` and ``known`` are ``(..., A)`` arrays; the result has shape ``S.shape[:-1]``.
    Attributes missing on either side are dropped and the weights renormalised.
    Pairs with no co-present attribute score 0.
    """
    w = weights.astype(np.float64)
    effective = known.astype(np.float64) * w
    denom = effective.sum(axis=-1)
    numer = (S.astype(np.float64) * effective).sum(axis=-1)
    out = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    return out


def _group_mean(
    S: np.ndarray, known: np.ndarray, weights: np.ndarray, *, penalise_missing: bool
) -> np.ndarray:
    """Weighted mean of one attribute group.

    With ``penalise_missing`` an attribute missing on either side keeps its
    weight in the denominator (so gaps lower the score); otherwise missing
    attributes are dropped and the weights renormalised (available-case).
    """
    w = weights.astype(np.float64)
    known_f = known.astype(np.float64)
    numer = (S.astype(np.float64) * w * known_f).sum(axis=-1)
    denom = (w * (np.ones_like(known_f) if penalise_missing else known_f)).sum(axis=-1)
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)


def clinical_columns(
    attribute_names: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Column indices of the clinical and auxiliary attribute groups."""
    clinical = [i for i, n in enumerate(attribute_names) if n in CLINICAL_ATTRIBUTES]
    auxiliary = [i for i, n in enumerate(attribute_names) if n not in CLINICAL_ATTRIBUTES]
    return clinical, auxiliary


def composite_similarities(
    S: np.ndarray,
    known: np.ndarray,
    weights: np.ndarray,
    attribute_names: Sequence[str],
    *,
    clinical_alpha: float = 0.7,
) -> np.ndarray:
    """Clinical-first composite global similarity.

    ``clinical_alpha`` is the share of the score carried by the clinical group
    (``injuries`` + ``losses``, missing-penalised); the rest is the auxiliary
    group (available-case).  When only one group is present in
    ``attribute_names`` it carries the whole score.  ``S``/``known`` are
    ``(..., A)``; the result has shape ``S.shape[:-1]``.
    """
    S = np.asarray(S, dtype=np.float64)
    known = np.asarray(known, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    clinical, auxiliary = clinical_columns(attribute_names)
    if not clinical:
        return _group_mean(S, known, w, penalise_missing=False)
    if not auxiliary:
        return _group_mean(S, known, w, penalise_missing=True)
    clinical_part = _group_mean(
        S[..., clinical], known[..., clinical], w[clinical], penalise_missing=True
    )
    auxiliary_part = _group_mean(
        S[..., auxiliary], known[..., auxiliary], w[auxiliary],
        penalise_missing=False,
    )
    return clinical_alpha * clinical_part + (1.0 - clinical_alpha) * auxiliary_part


def clinical_gate_mask(
    S: np.ndarray,
    known: np.ndarray,
    attribute_names: Sequence[str],
    *,
    min_injury_similarity: float | None,
    query_has_injuries: bool | np.ndarray,
) -> np.ndarray:
    """Mask of candidates allowed past the loose clinical gate.

    When the query has injuries, a candidate must carry an injury set whose
    graded similarity clears ``min_injury_similarity``.  The floor is
    deliberately low: it exists to stop treatment/count agreement buying a
    clinically unrelated case into the top-k, not to demand an exact ICD match.
    A query without injuries, or a ``None``/non-positive threshold, passes
    everyone.
    """
    shape = S.shape[:-1]
    if (
        "injuries" not in attribute_names
        or min_injury_similarity is None
        or min_injury_similarity <= 0
    ):
        return np.ones(shape, dtype=bool)
    idx = list(attribute_names).index("injuries")
    mask = known[..., idx] & (S[..., idx] >= min_injury_similarity)
    qhi = np.asarray(query_has_injuries, dtype=bool)
    if qhi.ndim == 0:
        return mask if bool(qhi) else np.ones(shape, dtype=bool)
    return np.where(qhi[..., None], mask, True)


class CorpusCBRModel:
    """A myCBR model over a list of :class:`CorpusCase` records."""

    def __init__(
        self,
        cases: Sequence[CorpusCase],
        weights: dict[str, float] | None = None,
        taxonomy: IcdTaxonomy | None = None,
        attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
        missing_policy: MissingPolicy = "available",
    ) -> None:
        if not cases:
            raise ValueError("Cannot build a CBR model from an empty corpus")
        unknown = [a for a in attributes if a not in ALL_ATTRIBUTES]
        if unknown:
            raise ValueError(
                f"Unknown attributes: {unknown}. Known: {ALL_ATTRIBUTES}"
            )
        self.cases: list[CorpusCase] = list(cases)
        self.attribute_names: tuple[str, ...] = tuple(attributes)
        self.missing_policy: MissingPolicy = missing_policy
        self._weights: dict[str, float] = {
            name: float((weights or {}).get(name, 1.0))
            for name in self.attribute_names
        }
        self._taxonomy = taxonomy or IcdTaxonomy.from_json()

        self.prj, self.concept, self.cb = self._build_project()
        self._instances = self._populate_case_base()

        amalgam = self.concept.get_active_amalgam_fct()
        assert amalgam is not None
        self._amalgam = amalgam
        self._descs: dict[str, AttributeDesc] = {
            name: self.concept.get_attribute_desc(name)  # type: ignore[misc]
            for name in self.attribute_names
        }

    # -- construction -----------------------------------------------------
    def _build_project(self) -> tuple[Project, Concept, DefaultCaseBase]:
        prj = Project()
        concept = prj.create_top_concept(CONCEPT_ID)

        injury_codes = sorted({code for c in self.cases for code in c.injuries})

        # Symbol / multiple-symbol descriptions.
        if "gender" in self.attribute_names:
            SymbolDesc(concept, "gender", GENDER_VALUES)
        if "injuries" in self.attribute_names:
            injuries_desc = SymbolDesc(concept, "injuries", injury_codes)
            injuries_desc.is_multiple = True
            icd_fct = IcdCodeSimFct(prj, injuries_desc, self._taxonomy)
            icd_fct.mc = MaxOverlapMultipleConfig()
            injuries_desc.add_function(icd_fct, active=True)
        if "losses" in self.attribute_names:
            losses_desc = SymbolDesc(concept, "losses", ALL_LOSS_CATEGORIES)
            losses_desc.is_multiple = True
            losses_fct = losses_desc.get_fct(Project.DEFAULT_FCT_NAME)
            if not isinstance(losses_fct, SymbolFct):
                losses_fct = losses_desc.add_symbol_fct(Project.DEFAULT_FCT_NAME, True)
            losses_fct.mc = JaccardMultipleConfig()
        if "overall_category" in self.attribute_names:
            severity_desc = SymbolDesc(concept, "overall_category", SEVERITY_VALUES)
            severity_fct = severity_desc.get_fct(Project.DEFAULT_FCT_NAME)
            if not isinstance(severity_fct, SymbolFct):
                severity_fct = severity_desc.add_symbol_fct(
                    Project.DEFAULT_FCT_NAME, True
                )
            severity_fct.is_symmetric = True
            last = len(SEVERITY_VALUES) - 1
            for i, vi in enumerate(SEVERITY_VALUES):
                for j, vj in enumerate(SEVERITY_VALUES):
                    severity_fct.set_similarity(
                        vi, vj, 1.0 - abs(i - j) / last
                    )

        # Integer descriptions.
        integer_specs = {
            "age_at_accident": (AGE_MIN, AGE_MAX),
            "age_at_trial": (AGE_MIN, AGE_MAX),
            "operations_count": (0, OPS_MAX),
            "hospitalisation_days": (0, HOSP_MAX),
            "n_injuries": (0, N_INJ_MAX),
            "n_losses": (0, N_LOSS_MAX),
        }
        numeric_descs: list[IntegerDesc] = []
        for name, (low, high) in integer_specs.items():
            if name in self.attribute_names:
                numeric_descs.append(IntegerDesc(concept, name, low, high))

        psla_desc = DoubleDesc(concept, "psla_amount", PSLA_MIN, PSLA_MAX)
        psla_desc.is_solution = True

        amalgam = concept.get_active_amalgam_fct()
        assert amalgam is not None
        amalgam.set_type(AmalgamationConfig.WEIGHTED_SUM)
        amalgam.set_active(psla_desc, False)

        for desc in numeric_descs:
            fct = desc.add_integer_fct(f"{desc.name}_sim", False)
            fct.set_function_type_r(NumberConfig.POLYNOMIAL_WITH)
            fct.set_function_parameter_r(2.0)
            fct.set_function_type_l(NumberConfig.POLYNOMIAL_WITH)
            fct.set_function_parameter_l(2.0)
            amalgam.set_active_fct(desc, fct)

        for name, weight in self._weights.items():
            amalgam.set_weight(name, weight)

        cb = prj.create_default_cb(CASE_BASE_NAME)
        return prj, concept, cb

    def _populate(
        self, inst: Instance, case: CorpusCase, *, use_award_severity: bool = False
    ) -> None:
        if "gender" in self.attribute_names and case.gender:
            inst.add_attribute("gender", case.gender)
        if "age_at_accident" in self.attribute_names and case.age_at_accident is not None:
            inst.add_attribute("age_at_accident", case.age_at_accident)
        if "age_at_trial" in self.attribute_names and case.age_at_trial is not None:
            inst.add_attribute("age_at_trial", case.age_at_trial)
        if "overall_category" in self.attribute_names:
            # Candidates use the award band (authoritative); a query uses the
            # judge/extractor label, which is never derived from its own award.
            severity = (
                case.award_severity if use_award_severity else case.overall_category
            )
            if severity:
                inst.add_attribute("overall_category", severity)

        for name, value in (
            ("operations_count", case.operations_count),
            ("hospitalisation_days", case.hospitalisation_days),
            ("n_injuries", case.n_injuries),
            ("n_losses", case.n_losses),
        ):
            if name in self.attribute_names and value is not None:
                inst.add_attribute(name, value)

        if "injuries" in self.attribute_names:
            desc = self.concept.get_attribute_desc("injuries")
            assert isinstance(desc, SymbolDesc)
            injuries = _multiple(desc, case.injuries)
            if injuries is not None:
                inst.add_attribute(desc, injuries)

        if "losses" in self.attribute_names:
            desc = self.concept.get_attribute_desc("losses")
            assert isinstance(desc, SymbolDesc)
            losses = _multiple(desc, case.losses)
            if losses is not None:
                inst.add_attribute(desc, losses)

        if case.psla_real is not None:
            inst.add_attribute("psla_amount", case.psla_real)

    def _populate_case_base(self) -> list[Instance]:
        instances: list[Instance] = []
        for case in self.cases:
            inst = self.concept.create_instance(case.case_id)
            self._populate(inst, case, use_award_severity=True)
            self.cb.add_case(inst)
            instances.append(inst)
        return instances

    # -- queries ----------------------------------------------------------
    def make_query(self, case: CorpusCase) -> Instance:
        query = self.concept.get_query_instance()
        self._populate(query, case, use_award_severity=False)
        return query

    # -- retrieval --------------------------------------------------------
    @property
    def candidate_ids(self) -> list[str]:
        return [c.case_id for c in self.cases]

    def _instance_by_id(self) -> dict[str, Instance]:
        return {inst.name: inst for inst in self._instances}

    def _set_weights(self, weights: dict[str, float] | None) -> dict[str, float]:
        old = {name: self._amalgam.weights[name] for name in self.attribute_names}
        if weights:
            for name in self.attribute_names:
                self._amalgam.set_weight(name, float(weights.get(name, 1.0)))
        return old

    def _weight_vector(self, weights: dict[str, float] | None) -> np.ndarray:
        merged = dict(self._weights)
        if weights:
            merged.update({k: float(v) for k, v in weights.items()})
        return np.asarray(
            [merged.get(name, 1.0) for name in self.attribute_names],
            dtype=np.float64,
        )

    def retrieve(
        self,
        query: CorpusCase,
        k: int | None = None,
        weights: dict[str, float] | None = None,
        exclude: str | None = None,
        candidate_ids: Sequence[str] | None = None,
        *,
        clinical_alpha: float = 0.7,
        min_injury_similarity: float | None = None,
    ) -> list[tuple[str, float]]:
        """CBR retrieval for one query, returning ``[(case_id, similarity), ...]``.

        Scoring is the clinical-first composite (clinical group missing-penalised,
        auxiliary group available-case) with an optional loose injury-similarity
        gate.  Set ``missing_policy="special"`` for the raw myCBR amalgamation.
        """
        if self.missing_policy == "special":
            return self._retrieve_special(
                query, k=k, weights=weights, exclude=exclude,
                candidate_ids=candidate_ids,
            )

        query_inst = self.make_query(query)
        if candidate_ids is None:
            instances = self._instances
        else:
            allowed = set(candidate_ids)
            instances = [i for i in self._instances if i.name in allowed]

        active = self.active_fcts()
        S = np.empty((len(instances), len(active)), dtype=np.float64)
        known = np.empty((len(instances), len(active)), dtype=bool)
        for ci, inst in enumerate(instances):
            for ai, (_name, desc, fct) in enumerate(active):
                q_att = query_inst.get_att_for_desc(desc)
                c_att = inst.get_att_for_desc(desc)
                S[ci, ai] = fct.calculate_similarity(q_att, c_att).value
                known[ci, ai] = not (
                    isinstance(q_att, SpecialAttribute)
                    or isinstance(c_att, SpecialAttribute)
                )
        sims = composite_similarities(
            S, known, self._weight_vector(weights), self.attribute_names,
            clinical_alpha=clinical_alpha,
        )
        gate = clinical_gate_mask(
            S, known, self.attribute_names,
            min_injury_similarity=min_injury_similarity,
            query_has_injuries=bool(query.injuries),
        )

        scored = [
            (inst.name, float(sims[ci]))
            for ci, inst in enumerate(instances)
            if gate[ci] and not (exclude is not None and inst.name == exclude)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k] if k is not None else scored

    def _retrieve_special(
        self,
        query: CorpusCase,
        k: int | None = None,
        weights: dict[str, float] | None = None,
        exclude: str | None = None,
        candidate_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        query_inst = self.make_query(query)
        if candidate_ids is None:
            instances = self._instances
        else:
            allowed = set(candidate_ids)
            instances = [i for i in self._instances if i.name in allowed]

        old = self._set_weights(weights)
        try:
            scored: list[tuple[str, float]] = []
            for inst in instances:
                if exclude is not None and inst.name == exclude:
                    continue
                sim = self._amalgam.calculate_similarity(query_inst, inst).value
                scored.append((inst.name, float(sim)))
        finally:
            self._amalgam.weights.update(old)
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k] if k is not None else scored

    def retrieve_cbr(
        self,
        query: CorpusCase,
        k: int | None = None,
        candidate_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Reference retrieval via myCBR's :class:`SequentialRetrieval`."""
        query_inst = self.make_query(query)
        if candidate_ids is None:
            ranked = SequentialRetrieval(self.prj).retrieve_sorted(self.cb, query_inst)
        else:
            from hklii_psla.cbr.core.default_case_base import DefaultCaseBase

            sub = DefaultCaseBase(self.prj, "subset")
            allowed = set(candidate_ids)
            for inst in self._instances:
                if inst.name in allowed:
                    sub.add_case(inst)
            ranked = SequentialRetrieval(self.prj).retrieve_sorted(sub, query_inst)
        out = [(inst.name, float(sim.value)) for inst, sim in ranked]
        return out[:k] if k is not None else out

    # -- per-attribute similarity cache ----------------------------------
    def active_fcts(self) -> list[tuple[str, AttributeDesc, object]]:
        out: list[tuple[str, AttributeDesc, object]] = []
        for name in self.attribute_names:
            desc = self._descs[name]
            fct = self._amalgam.get_active_fct(desc)
            assert fct is not None, f"no active fct for {name}"
            out.append((name, desc, fct))
        return out

    def per_attribute_similarities(
        self,
        queries: Sequence[CorpusCase],
        candidate_ids: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-attribute similarities ``(Q, C, A)`` and presence mask ``(Q, C, A)``.

        ``A`` follows :attr:`attribute_names`; ``C`` follows ``candidate_ids``
        (defaults to :attr:`candidate_ids`).  ``known[q, c, a]`` is ``True`` when
        both sides carry a non-special value for attribute ``a``.
        """
        if candidate_ids is None:
            instances = self._instances
        else:
            by_id = self._instance_by_id()
            instances = [by_id[cid] for cid in candidate_ids]
        active = self.active_fcts()

        S = np.empty((len(queries), len(instances), len(active)), dtype=np.float32)
        known = np.empty((len(queries), len(instances), len(active)), dtype=bool)
        for qi, case in enumerate(queries):
            query_inst = self.make_query(case)
            for ci, inst in enumerate(instances):
                for ai, (_name, desc, fct) in enumerate(active):
                    q_att = query_inst.get_att_for_desc(desc)
                    c_att = inst.get_att_for_desc(desc)
                    S[qi, ci, ai] = fct.calculate_similarity(q_att, c_att).value
                    known[qi, ci, ai] = not (
                        isinstance(q_att, SpecialAttribute)
                        or isinstance(c_att, SpecialAttribute)
                    )
        return S, known


def build_corpus_model(
    cases: Sequence[CorpusCase],
    weights: dict[str, float] | None = None,
    taxonomy: IcdTaxonomy | None = None,
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    missing_policy: MissingPolicy = "available",
) -> CorpusCBRModel:
    return CorpusCBRModel(
        cases,
        weights=weights,
        taxonomy=taxonomy,
        attributes=attributes,
        missing_policy=missing_policy,
    )
