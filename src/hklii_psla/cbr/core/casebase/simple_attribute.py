from __future__ import annotations

from hklii_psla.cbr.core.casebase.attribute import Attribute
from hklii_psla.cbr.core.explanation.explainable import Explainable, ExplainableType
from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc

from typing import override

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
