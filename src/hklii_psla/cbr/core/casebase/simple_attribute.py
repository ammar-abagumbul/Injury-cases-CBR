from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.attribute import Attribute
from hklii_psla.cbr.core.explanation.explainable import Explainable, ExplainableType

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc


class SimpleAttribute(Attribute, Explainable):

    def __init__(self, desc: AttributeDesc):
        self._desc = desc

    @property
    @override
    def name(self) -> str:
        return self.get_value_as_string()

    @property
    @override
    def exp_type(self) -> ExplainableType:
        return ExplainableType.SIMPLE_ATTRIBUTE

    @override
    def get_value_as_string(self) -> str:
        raise NotImplementedError("get_value_as_string not implemented")

    @property
    def attribute_desc(self) -> AttributeDesc:
        return self._desc

    @property
    def desc(self) -> AttributeDesc:
        return self._desc

    @desc.setter
    def desc(self, value: AttributeDesc) -> None:
        self._desc = value
