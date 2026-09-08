from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.attribute import Attribute
from hklii_psla.cbr.core.explanation.explainable import Explainable, ExplainableType

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.concept import Concept


class Instance(Attribute, Explainable):

    QUERY_ID = "query"

    def __init__(self, concept: Concept, id_: str):
        self._concept = concept
        self._id = id_
        self._attributes: dict[AttributeDesc, Attribute] = {}
        self._adapted = False
        self.reset()
        if id_ != "query":
            concept.add_instance(self)

    def set_name(self, n: str) -> None:
        if not self._id or not self._id.strip() or self._concept.project.get_instance(self._id) is not None:
            return
        self._concept.rename_instance(self._id, n)
        self._id = n

    @property
    @override
    def name(self) -> str:
        return self._id

    @property
    def concept(self) -> Concept:
        return self._concept

    @property
    def attributes(self) -> dict[AttributeDesc, Attribute]:
        return self._attributes

    def remove_attribute(self, desc: AttributeDesc) -> None:
        self._attributes.pop(desc, None)

    def add_attribute(self, desc_or_name, value) -> bool:
        if isinstance(desc_or_name, str):
            att_desc = self._concept.get_all_attribute_descs().get(desc_or_name)
        else:
            att_desc = desc_or_name

        if att_desc is None:
            return False

        if not isinstance(value, Attribute):
            value = att_desc.get_attribute(value)

        if value is None:
            return False

        if att_desc.fits(value):
            self._attributes[att_desc] = value
            return True
        return False

    def get_att_for_desc(self, att_desc: AttributeDesc) -> Attribute | None:
        return self._attributes.get(att_desc)

    @override
    def get_value_as_string(self) -> str:
        return self._id

    @property
    @override
    def exp_type(self) -> ExplainableType:
        return ExplainableType.INSTANCE

    def clean(self, desc: AttributeDesc | None = None) -> None:
        if desc is not None:
            att = self.get_att_for_desc(desc)
            if not desc.fits(att):
                self.add_attribute(desc, self._concept.project.get_special_attribute(
                    self._concept.project.UNKNOWN_SPECIAL_VALUE
                ))
            return

        allowed_descs = set(self._concept.get_all_attribute_descs().values())
        for old_desc in [d for d in self._attributes if d not in allowed_descs]:
            del self._attributes[old_desc]

        for att_desc in self._concept.get_all_attribute_descs().values():
            if self.get_att_for_desc(att_desc) is None:
                self.add_attribute(
                    att_desc,
                    self._concept.project.get_special_attribute(self._concept.project.UNKNOWN_SPECIAL_VALUE),
                )

    def reset(self) -> None:
        self._attributes.clear()
        sv_att = self._concept.project.get_special_attribute(self._concept.project.UNKNOWN_SPECIAL_VALUE)
        for desc in self._concept.get_all_attribute_descs().values():
            self._attributes[desc] = sv_att

    def set_atts_unknown(self) -> None:
        self.reset()

    def __str__(self) -> str:
        return self._id

    @property
    def adapted(self) -> bool:
        return self._adapted

    @adapted.setter
    def adapted(self, value: bool) -> None:
        self._adapted = value
