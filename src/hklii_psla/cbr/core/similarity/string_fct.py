from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig
from hklii_psla.cbr.core.similarity.config.string_config import StringConfig
from hklii_psla.cbr.core.similarity.sim_fct import SimFct
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.string_desc import StringDesc
    from hklii_psla.cbr.core.project import Project


class StringFct(SimFct):

    def __init__(self, prj: Project, config: StringConfig, desc: StringDesc, name: str):
        super().__init__(prj, desc, name)
        self.sub_desc = desc
        self.config = config
        self.n = 3
        self.levenshtein_del_cost = 1
        self.levenshtein_add_cost = 1
        self.levenshtein_change_cost = 1
        self.case_sensitive = True
        self.mc = MultipleConfig.DEFAULT_CONFIG

    @property
    @override
    def is_symmetric(self) -> bool:
        return True

    @is_symmetric.setter
    @override
    def is_symmetric(self, value: bool) -> None:
        pass

    @override
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity:
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
        from hklii_psla.cbr.core.casebase.string_attribute import StringAttribute

        if isinstance(a1, MultipleAttribute) and isinstance(a2, MultipleAttribute):
            return self.prj.calculate_multiple_attribute_similarity(self, a1, a2)
        if isinstance(a1, SpecialAttribute) or isinstance(a2, SpecialAttribute):
            return self.prj.calculate_special_similarity(a1, a2)

        assert isinstance(a1, StringAttribute) and isinstance(a2, StringAttribute)
        v1, v2 = str(a1), str(a2)
        if not self.case_sensitive:
            v1, v2 = v1.lower(), v2.lower()

        if self.config == StringConfig.EQUALITY:
            return Similarity.get(1.00) if v1 == v2 else Similarity.get(0.00)
        if self.config == StringConfig.NGRAM:
            return self._ngram_similarity(v1, v2)
        if self.config == StringConfig.LEVENSHTEIN:
            return self._levenshtein_similarity(v1, v2)
        return Similarity.INVALID_SIM

    def _ngram_similarity(self, v1: str, v2: str) -> Similarity:
        n = self.n
        ngrams1 = {v1[i:i + n] for i in range(len(v1) - n + 1)}
        ngrams2 = {v2[i:i + n] for i in range(len(v2) - n + 1)}
        union = ngrams1 | ngrams2
        if not union:
            return Similarity.get(0.0)
        common = ngrams1 & ngrams2
        return Similarity.get(len(common) / len(union))

    def _levenshtein_similarity(self, s0: str, s1: str) -> Similarity:
        len0, len1 = len(s0) + 1, len(s1) + 1
        cost = list(range(len0))
        newcost = [0] * len0

        for j in range(1, len1):
            newcost[0] = j
            for i in range(1, len0):
                match = 0 if s0[i - 1] == s1[j - 1] else self.levenshtein_change_cost
                cost_replace = cost[i - 1] + match
                cost_insert = cost[i] + self.levenshtein_add_cost
                cost_delete = newcost[i - 1] + self.levenshtein_del_cost
                newcost[i] = min(cost_insert, cost_delete, cost_replace)
            cost, newcost = newcost, cost

        max_length = max(len(s0), len(s1))
        if max_length == 0:
            return Similarity.get(1.0)
        result = 1 - (cost[len0 - 1] / max_length)
        return Similarity.get(result)

    def get_multiple_config(self) -> MultipleConfig:
        return self.mc

    def set_multiple_config(self, mc: MultipleConfig) -> None:
        self.mc = mc

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.string_desc import StringDesc
        from hklii_psla.cbr.core.project import Project

        if isinstance(new_desc, StringDesc) and self.name != Project.DEFAULT_FCT_NAME:
            f = new_desc.add_string_fct(self.config, self.name, is_active)
            f.case_sensitive = self.case_sensitive
            f.mc = self.mc
            f.n = self.n
