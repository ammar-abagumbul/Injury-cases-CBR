from typing_extensions import Collection, override

from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
from hklii_psla.cbr.core.explanation.symbol_desc import SymbolDesc
from hklii_psla.cbr.core.similarity.taxonomy_node import TaxonomyNode


class SymbolAttribute(SimpleAttribute, TaxonomyNode):

    def __init__(
        self,
        desc: SymbolDesc,
        v: str
    ):
        super().__init__(desc)
        self._value = v
        self._desc: SymbolDesc = desc

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, v: str) -> None:
        if self._desc.is_allowed_value(v):
            return
        if self._desc.rename_value(self._value, v):
            self._value = v
            # TODO
            # set_changed()
            # notify_observers()
        self._value = v

    @property
    @override
    def nodes(self) -> list[TaxonomyNode]:
        return list(self.get_symbol_attrs())
