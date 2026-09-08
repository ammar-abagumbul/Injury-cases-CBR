from __future__ import annotations

from typing import TYPE_CHECKING

from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
    from hklii_psla.cbr.core.model.special_desc import SpecialDesc
    from hklii_psla.cbr.core.project import Project


class SpecialFct(SymbolFct):
    """Computes similarity of special values ("_undefined_"/"_unknown_"/"_others_"),
    treated like ordinary symbol attributes with a lookup table."""

    def __init__(self, prj: Project, desc: SpecialDesc, name: str):
        super().__init__(prj, desc, name)

    def calculate_special_similarity(
        self, att1: SpecialAttribute | None, att2: SpecialAttribute | None
    ) -> Similarity:
        from hklii_psla.cbr.core.project import Project

        result = Similarity.INVALID_SIM
        if att1 is None:
            att1 = self.prj.get_special_attribute(Project.UNDEFINED_SPECIAL_ATTRIBUTE)
        if att2 is None:
            att2 = self.prj.get_special_attribute(Project.UNDEFINED_SPECIAL_ATTRIBUTE)

        if att1.attribute_desc is not None and att1.attribute_desc == att2.attribute_desc:
            index1 = self.desc.get_index_of(att1)  # type: ignore[attr-defined]
            index2 = self.desc.get_index_of(att2)  # type: ignore[attr-defined]
            if index1 is not None and index2 is not None:
                try:
                    result = self.sims[index1][index2]
                except IndexError:
                    pass
        return result
