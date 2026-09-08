from __future__ import annotations

from typing import TYPE_CHECKING

from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.special_desc import SpecialDesc
    from hklii_psla.cbr.core.project import Project
    from hklii_psla.cbr.core.similarity.similarity import Similarity


class SpecialAttribute(SymbolAttribute):
    """Wraps a special value ("_undefined_", "_unknown_", "_others_") so it can be
    used as a value for any attribute description."""

    def __init__(self, project: Project, desc: SpecialDesc, v: str):
        super().__init__(desc, v)
        self._project = project

    def calculate_similarity(self, att: SymbolAttribute) -> Similarity:
        return self._project.calculate_special_similarity(self, att)
