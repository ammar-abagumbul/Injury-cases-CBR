from __future__ import annotations

import math
from typing import TYPE_CHECKING

from hklii_psla.cbr.core.similarity.config.amalgamation_config import AmalgamationConfig
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.project import Project
    from hklii_psla.cbr.core.similarity.sim_fct import SimFctInterface


class AmalgamationFct:
    """Computes the similarity of two instances of a concept by combining
    the similarity of their individual attributes (minimum, maximum,
    euclidean, or weighted sum)."""

    def __init__(self, amalgamation_type: AmalgamationConfig, concept: Concept, name: str):
        self.type = amalgamation_type
        self.concept = concept
        self.name = name
        self.weights: dict[str, float] = {}
        self.active: dict[AttributeDesc, bool] = {}
        self.active_fcts: dict[AttributeDesc, object] = {}

        for att in concept.get_all_attribute_descs().values():
            if att.name not in self.weights:
                self.weights[att.name] = 1.0
                self.active[att] = True
                from hklii_psla.cbr.core.model.concept_desc import ConceptDesc
                from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc

                if not isinstance(att, ConceptDesc) and isinstance(att, SimpleAttDesc):
                    fcts = att.sim_fcts
                    if fcts:
                        self.active_fcts[att] = fcts[0]
                elif isinstance(att, ConceptDesc):
                    fcts = att.concept.get_available_amalgam_fcts()
                    if fcts:
                        self.active_fcts[att] = fcts[0]

    def calculate_similarity(self, value1: Attribute, value2: Attribute) -> Similarity:
        from hklii_psla.cbr.core.casebase.instance import Instance
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if isinstance(value1, SpecialAttribute) or isinstance(value2, SpecialAttribute):
            return self.concept.project.calculate_special_similarity(value1, value2)
        if isinstance(value1, MultipleAttribute) and isinstance(value2, MultipleAttribute):
            return self.concept.project.calculate_multiple_attribute_similarity(self, value1, value2)

        assert isinstance(value1, Instance) and isinstance(value2, Instance)
        max_sim = Similarity.get(0.0)
        min_sim = Similarity.get(1.0)
        sims: dict[AttributeDesc, Similarity] = {}
        normalize = 0.0

        for att_desc in self.concept.get_all_attribute_descs().values():
            if self.active.get(att_desc):
                q_att = value1.get_att_for_desc(att_desc)
                c_att = value2.get_att_for_desc(att_desc)
                f = self.active_fcts.get(att_desc)

                sim = Similarity.INVALID_SIM
                if f is not None:
                    if isinstance(f, AmalgamationFct):
                        if q_att is not None:
                            sim = f.calculate_similarity(q_att, c_att)
                    else:
                        sim = f.calculate_similarity(q_att, c_att)  # type: ignore[union-attr]

                    weight = self.weights[att_desc.name]
                    tmp = sim.value * weight
                    if tmp < min_sim.value:
                        min_sim = Similarity.get(tmp)
                    if tmp > max_sim.value:
                        max_sim = Similarity.get(tmp)
                    sims[att_desc] = sim
                    normalize += weight
            else:
                sims[att_desc] = Similarity.get(0.00)

        result = -1.0
        if self.type == AmalgamationConfig.MAXIMUM:
            result = max_sim.value / normalize if normalize else 0.0
        elif self.type == AmalgamationConfig.MINIMUM:
            result = min_sim.value / normalize if normalize else 0.0
        elif self.type == AmalgamationConfig.EUCLIDEAN:
            tmp = 0.0
            for desc, sim in sims.items():
                if self.active.get(desc):
                    tmp += (self.weights[desc.name] / normalize) * (sim.value ** 2) if normalize else 0.0
            result = math.sqrt(tmp)
        elif self.type == AmalgamationConfig.WEIGHTED_SUM:
            tmp = 0.0
            norm = 0.0
            for desc, sim in sims.items():
                if self.active.get(desc):
                    weight = self.weights[desc.name]
                    tmp += weight * sim.value
                    norm += weight
            if tmp != 0.0 and norm != 0.0:
                tmp /= norm
            result = tmp
        elif self.type == AmalgamationConfig.SIM_DEF:
            result = 0.0
        else:
            raise ValueError(f"AmalgamationConfig value unknown: {self.type}")

        if result == -1.0:
            return Similarity.INVALID_SIM
        return Similarity.get(result)

    def get_type(self) -> AmalgamationConfig:
        return self.type

    def set_type(self, amalgamation_type: AmalgamationConfig) -> None:
        self.type = amalgamation_type

    def set_weight(self, name_or_desc, weight: float) -> None:
        name = name_or_desc if isinstance(name_or_desc, str) else name_or_desc.name
        self.weights[name] = weight

    def get_weight(self, desc: AttributeDesc) -> float | None:
        return self.weights.get(desc.name)

    @property
    def project(self) -> Project:
        return self.concept.project

    def is_active(self, att: AttributeDesc) -> bool | None:
        return self.active.get(att)

    def set_active(self, att: AttributeDesc, active: bool) -> None:
        self.active[att] = active

    def set_active_fct(self, att: AttributeDesc, active_sim: object) -> None:
        from hklii_psla.cbr.core.similarity.sim_fct import SimFctInterface

        if isinstance(active_sim, (SimFctInterface, AmalgamationFct)):
            self.active_fcts[att] = active_sim

    def get_active_fct(self, att: AttributeDesc) -> object | None:
        return self.active_fcts.get(att)

    def remove(self, desc: AttributeDesc) -> None:
        self.weights.pop(desc.name, None)
        self.active.pop(desc, None)
        self.active_fcts.pop(desc, None)
