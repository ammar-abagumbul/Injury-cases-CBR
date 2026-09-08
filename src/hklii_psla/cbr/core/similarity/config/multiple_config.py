from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
    from hklii_psla.cbr.core.similarity.similarity import Similarity


class MainType(str, Enum):
    BEST_MATCH = "BEST_MATCH"
    WORST_MATCH = "WORST_MATCH"
    PARTNER_MAX = "PARTNER_MAX"
    PARTNER_CASE = "PARTNER_CASE"
    PARTNER_QUERY = "PARTNER_QUERY"


class Reuse(str, Enum):
    REUSE = "REUSE"
    ZERO_SIM = "ZERO_SIM"
    IGNORE = "IGNORE"
    NONE = "NONE"


class Type(str, Enum):
    AVG = "AVG"
    MAX = "MAX"
    MIN = "MIN"
    NONE = "NONE"


class _InnerFct(Protocol):
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity: ...


class MultipleConfig:
    """Configures how to compute similarity between two MultipleAttribute values."""

    DEFAULT_CONFIG: MultipleConfig

    def __init__(self, main_type: MainType, reuse: Reuse, type_: Type):
        self.main_type = main_type
        self.reuse = reuse
        self.type = type_

    def calculate_similarity(
        self,
        inner_fct: _InnerFct,
        value1: MultipleAttribute,
        value2: MultipleAttribute,
    ) -> Similarity:
        from hklii_psla.cbr.core.similarity.similarity import Similarity

        if inner_fct is None or value1 is None or value2 is None:
            return Similarity.INVALID_SIM

        if self.main_type is MainType.BEST_MATCH:
            return self._calculate_best_match(inner_fct, value1, value2)
        if self.main_type is MainType.WORST_MATCH:
            return self._calculate_worst_match(inner_fct, value1, value2)
        if self.main_type is MainType.PARTNER_CASE:
            return self._calculate_partner_sim(inner_fct, value2, value1)
        if self.main_type is MainType.PARTNER_QUERY:
            return self._calculate_partner_sim(inner_fct, value1, value2)
        if self.main_type is MainType.PARTNER_MAX:
            if len(value1.values) >= len(value2.values):
                return self._calculate_partner_sim(inner_fct, value1, value2)
            return self._calculate_partner_sim(inner_fct, value2, value1)
        return Similarity.INVALID_SIM

    def _calculate_partner_sim(
        self,
        inner_fct: _InnerFct,
        value1: MultipleAttribute,
        value2: MultipleAttribute,
    ) -> Similarity:
        from hklii_psla.cbr.core.similarity.similarity import Similarity

        is_reuse = self.reuse is Reuse.REUSE
        sims: list[float] = []
        used_items: list[Attribute] = []

        for current_query_value in value1.values:
            if not is_reuse and len(used_items) == len(value2.values):
                if self.reuse is Reuse.ZERO_SIM:
                    sims.extend([0.0] * (len(value1.values) - len(sims)))
                break

            sim_max = -1.0
            att: Attribute | None = None
            for current_case_value in value2.values:
                tmp_sim = inner_fct.calculate_similarity(current_query_value, current_case_value).value
                if tmp_sim > sim_max and (is_reuse or current_case_value not in used_items):
                    sim_max = tmp_sim
                    att = current_case_value

            if att is not None:
                used_items.append(att)
            sims.append(sim_max)

        if not sims:
            return Similarity.INVALID_SIM
        if self.type is Type.AVG:
            return Similarity.get(sum(sims) / len(sims))
        if self.type is Type.MAX:
            return Similarity.get(max(0.0, max(sims)))
        if self.type is Type.MIN:
            return Similarity.get(min(1.0, min(sims)))
        return Similarity.INVALID_SIM

    def _calculate_worst_match(
        self, f: _InnerFct, value1: MultipleAttribute, value2: MultipleAttribute
    ) -> Similarity:
        from hklii_psla.cbr.core.similarity.similarity import Similarity

        result = Similarity.get(1.0)
        for att1 in value1.values:
            for att2 in value2.values:
                tmp_sim = f.calculate_similarity(att1, att2)
                if tmp_sim.value < result.value:
                    result = tmp_sim
                if result.value == 0.0:
                    return result
        return result

    def _calculate_best_match(
        self, f: _InnerFct, value1: MultipleAttribute, value2: MultipleAttribute
    ) -> Similarity:
        from hklii_psla.cbr.core.similarity.similarity import Similarity

        result = Similarity.INVALID_SIM
        for att1 in value1.values:
            for att2 in value2.values:
                tmp_sim = f.calculate_similarity(att1, att2)
                if tmp_sim.value > result.value:
                    result = tmp_sim
                if result.value == 1.0:
                    return result
        return result


MultipleConfig.DEFAULT_CONFIG = MultipleConfig(MainType.PARTNER_QUERY, Reuse.REUSE, Type.MAX)
