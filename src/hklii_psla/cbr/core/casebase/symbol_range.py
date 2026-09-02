from typing import final

from typing_extensions import Collection

from hklii_psla.cbr.core.casebase.range import Range
from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
from hklii_psla.cbr.core.explanation.symbol_desc import SymbolDesc
from hklii_psla.cbr.core.project import Project


class SymbolRange(Range):

    _symbols: dict[str, SymbolAttribute]

    def __init__(
        self,
        project: Project | None,
        description: SymbolDesc,
        allowed_values: Collection[str]
    ):
        super().__init__(project)
        self._desc: SymbolDesc = description

        if allowed_values:
            self.init_symbol_attrs([])
        else:
            self.init_symbol_attrs(allowed_values)

    def init_symbol_attrs(self, values: Collection[str]):
        self._symbols = {}
        for value in values:
            if value.strip():
                self._symbols[value] = SymbolAttribute(
                    self._desc,
                    value
                )


    @final
    def set_symbols(self):
        ...

    @final
    def rename_symbol(self, old_value: str, new_value: str):
        att = self._symbols.get(old_value)
        if att:
            self._symbols[new_value] = att
            del self._symbols[old_value]

    @property
    def symbols(self):
        return self._symbols
