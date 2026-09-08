from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig
from hklii_psla.cbr.core.similarity.sim_fct import SimFct
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
    from hklii_psla.cbr.core.project import Project


class SymbolFct(SimFct):
    """Holds the similarity of the known symbol attributes in a table
    indexed by the range's linear order."""

    def __init__(self, project: Project, desc: SymbolDesc, name: str):
        super().__init__(project, desc, name)
        self.sub_desc = desc
        self.mc = MultipleConfig.DEFAULT_CONFIG

        count = len(desc.get_allowed_values()) if desc is not None else 0
        size = count if count != 0 else 10
        self.sims: list[list[Similarity]] = [
            [Similarity.get(1.00) if i == j else Similarity.get(0.00) for j in range(size)]
            for i in range(size)
        ]

    @override
    def calculate_similarity(self, attribute: Attribute, attribute2: Attribute) -> Similarity:
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
        from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute

        if isinstance(attribute, SpecialAttribute) or isinstance(attribute2, SpecialAttribute):
            return self.prj.calculate_special_similarity(attribute, attribute2)
        if isinstance(attribute, MultipleAttribute) and isinstance(attribute2, MultipleAttribute):
            return self.prj.calculate_multiple_attribute_similarity(self, attribute, attribute2)
        if isinstance(attribute, SymbolAttribute) and isinstance(attribute2, SymbolAttribute):
            if attribute.attribute_desc == attribute2.attribute_desc:
                index1 = self.sub_desc.get_index_of(attribute)
                index2 = self.sub_desc.get_index_of(attribute2)
                if index1 is not None and index2 is not None:
                    try:
                        return self.sims[index1][index2]
                    except IndexError:
                        pass
        return Similarity.INVALID_SIM

    def calculate_similarity_by_name(self, value1: str, value2: str) -> Similarity:
        return self.calculate_similarity(self.desc.get_attribute(value1), self.desc.get_attribute(value2))

    def set_similarity(
        self,
        att1: SymbolAttribute | str,
        att2: SymbolAttribute | str,
        sim: Similarity | float,
    ) -> bool:
        from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute

        if isinstance(att1, str):
            att1 = self.sub_desc.get_attribute(att1)  # type: ignore[assignment]
        if isinstance(att2, str):
            att2 = self.sub_desc.get_attribute(att2)  # type: ignore[assignment]
        if not isinstance(sim, Similarity):
            sim = Similarity.get(sim)

        index1 = self.sub_desc.get_index_of(att1)  # type: ignore[arg-type]
        index2 = self.sub_desc.get_index_of(att2)  # type: ignore[arg-type]
        if index1 is not None and index2 is not None:
            try:
                self.sims[index1][index2] = sim
                if self.is_symmetric:
                    self.sims[index2][index1] = sim
                return True
            except IndexError:
                pass
        return False

    def add_attribute(self, att: SymbolAttribute) -> bool:
        index = self.sub_desc.get_index_of(att)
        if index is None:
            return False
        n = len(self.sims) + 1
        sims_new = [[Similarity.get(0.00) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if j == index:
                    sims_new[i][j] = Similarity.get(1.00) if i == j else Similarity.get(0.00)
                elif i == index:
                    sims_new[i][j] = Similarity.get(0.00)
                else:
                    old_i = i if i < index else i - 1
                    old_j = j if j < index else j - 1
                    sims_new[i][j] = self.sims[old_i][old_j]
        self.sims = sims_new
        return True

    def remove_attribute(self, index: int | None) -> bool:
        if index is None:
            return False
        n = len(self.sims) - 1
        if n < 0:
            return False
        sims_new = [[Similarity.get(0.00) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                src_i = i if i < index else i + 1
                src_j = j if j < index else j + 1
                sims_new[i][j] = self.sims[src_i][src_j]
        self.sims = sims_new
        return True

    @property
    @override
    def is_symmetric(self) -> bool:
        return self._is_symmetric

    @is_symmetric.setter
    @override
    def is_symmetric(self, value: bool) -> None:
        self._is_symmetric = value

    def get_desc(self) -> SymbolDesc:
        return self.sub_desc

    def get_multiple_config(self) -> MultipleConfig:
        return self.mc

    def set_multiple_config(self, mc: MultipleConfig) -> None:
        self.mc = mc

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc
        from hklii_psla.cbr.core.project import Project

        if type(new_desc) is type(self.desc) and self.name != Project.DEFAULT_FCT_NAME:
            assert isinstance(new_desc, SymbolDesc)
            f = new_desc.add_symbol_fct(self.name, is_active)
            f.sims = [row[:] for row in self.sims]
            f.is_symmetric = self.is_symmetric
            f.mc = self.mc
