from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
from hklii_psla.cbr.core.model.simple_att_desc import SimpleAttDesc
from hklii_psla.cbr.core.similarity.taxonomy_node import TaxonomyNode

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
    from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
    from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute
    from hklii_psla.cbr.core.casebase.symbol_range import SymbolRange
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.similarity.ordered_symbol_fct import OrderedSymbolFct
    from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct
    from hklii_psla.cbr.core.similarity.taxonomy_fct import TaxonomyFct


class SymbolDesc(SimpleAttDesc, TaxonomyNode):

    def __init__(
        self,
        owner: Concept,
        name: str,
        allowed_values: Collection[str],
    ):
        super().__init__(owner, name)
        from hklii_psla.cbr.core.casebase.symbol_range import SymbolRange

        self.range: SymbolRange = SymbolRange(owner.project, self, allowed_values)

        if owner is not None and owner is not owner.project:
            owner.add_attribute_desc(self)
        self.add_default_fct()

    @property
    def symbol_range(self) -> SymbolRange:
        return self.range  # type: ignore[return-value]

    def add_ordered_symbol_fct(self, name: str, active: bool) -> OrderedSymbolFct:
        from hklii_psla.cbr.core.similarity.ordered_symbol_fct import OrderedSymbolFct

        f = OrderedSymbolFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f

    def get_attribute(self, value: object) -> Attribute | None:
        assert isinstance(value, str)
        if self.owner.project.is_special_attribute(value):
            return self.owner.project.get_special_attribute(value)
        return self.symbol_range.get_attribute(value)

    def add_symbol(self, value: str) -> SymbolAttribute | None:
        return self.symbol_range.add_symbol_value(value)

    def get_symbol_attributes(self) -> Collection[SymbolAttribute]:
        return self.symbol_range.symbols.values()

    def get_allowed_values(self) -> set[str]:
        return set(self.symbol_range.symbols.keys())

    def is_allowed_value(self, value: str) -> bool:
        return value in self.symbol_range.symbols

    def remove_symbol(self, value: str) -> None:
        self.symbol_range.remove_attribute(value)
        self.owner.project.clean_instances(self.owner, self)

    @override
    def can_override(self, desc: AttributeDesc) -> bool:
        if isinstance(desc, SymbolDesc):
            allowed = desc.get_allowed_values()
            if allowed and allowed.issuperset(self.get_allowed_values()):
                return True
        return False

    def get_index_of(self, att: SimpleAttribute) -> int | None:
        return self.symbol_range.get_index_of(att)  # type: ignore[arg-type]

    def fits(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
        from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute

        if not super().fits(att):
            return False
        if isinstance(att, SymbolAttribute) and not isinstance(att, SpecialAttribute):
            return self._check(att)
        if isinstance(att, MultipleAttribute):
            for a in att.values:
                if not isinstance(a, SymbolAttribute) or not self._check(a):
                    return False
        return True

    def fits_single(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute
        from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute

        if not super().fits_single(att):
            return False
        if self.is_multiple:
            if isinstance(att, SymbolAttribute) and not isinstance(att, SpecialAttribute):
                return self._check(att)
            return False
        return self.fits(att)

    def _check(self, att: SymbolAttribute) -> bool:
        return self.is_allowed_value(att.value)

    def rename_value(self, old_value: str, new_value: str) -> bool:
        if new_value.strip():
            self.symbol_range.rename_symbol(old_value, new_value)
            return True
        return False

    def add_symbol_fct(self, name: str, active: bool) -> SymbolFct:
        from hklii_psla.cbr.core.similarity.symbol_fct import SymbolFct

        f = SymbolFct(self.owner.project, self, name)
        self.add_function(f, active)
        return f

    def add_taxonomy_fct(self, name: str, active: bool) -> TaxonomyFct:
        from hklii_psla.cbr.core.similarity.taxonomy_fct import TaxonomyFct

        f = TaxonomyFct(self.owner.project, self, list(self.get_symbol_attributes()), name)
        self.add_function(f, active)
        f.update_table()
        return f

    @override
    def add_default_fct(self) -> None:
        if self.owner is not None and self.owner is not self.owner.project:
            active_sim = self.add_symbol_fct(self.owner.project.DEFAULT_FCT_NAME, False)
            self.update_amalgamation_fcts(self.owner, active_sim)

    @property
    @override
    def nodes(self) -> list[TaxonomyNode]:
        return list(self.get_symbol_attributes())
