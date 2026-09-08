from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.casebase.range import Range
    from hklii_psla.cbr.core.model.concept import Concept


class AttributeDesc(ABC):

    DELETE_NOTIFICATION = "delete_notification"

    def __init__(self, owner: Concept, name: str):
        self._name = name
        self._owner = owner
        self._is_solution: bool = False
        self._is_multiple: bool = False
        self.range: Range

    @property
    def name(self) -> str:
        return self._name

    @property
    def owner(self) -> Concept:
        return self._owner

    @owner.setter
    def owner(self, value: Concept) -> None:
        self._owner = value

    @property
    def is_multiple(self) -> bool:
        return self._is_multiple

    @is_multiple.setter
    def is_multiple(self, value: bool) -> None:
        if value == self._is_multiple:
            return
        self._is_multiple = value
        if value:
            self._owner.set_all_instances_multiple(self)
        else:
            self._owner.set_all_instances_single(self)

    @property
    def is_solution(self) -> bool:
        return self._is_solution

    @is_solution.setter
    def is_solution(self, value: bool) -> None:
        self._is_solution = value

    @abstractmethod
    def can_override(self, desc: AttributeDesc) -> bool:
        ...

    @abstractmethod
    def delete_all_fcts(self) -> None:
        ...

    def get_attribute(self, value: object) -> Attribute | None:
        return self.range.get_attribute(value)

    def fits(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.instance import Instance
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if att is None:
            return False
        if isinstance(att, SpecialAttribute):
            if att.attribute_desc is None:
                return False
            if not att.attribute_desc.is_allowed_value(att.get_value_as_string()):  # type: ignore[attr-defined]
                return False
        elif isinstance(att, SimpleAttribute):
            if att.attribute_desc is not self or self.is_multiple:
                return False
        elif isinstance(att, Instance):
            if self.is_multiple:
                return False
        elif isinstance(att, MultipleAttribute):
            if not self.is_multiple:
                return False
        return True

    def fits_single(self, att: Attribute | None) -> bool:
        from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if not self.is_multiple:
            return self.fits(att)
        if att is None:
            return False
        if isinstance(att, SpecialAttribute):
            if att.attribute_desc is None:
                return False
            if not att.attribute_desc.is_allowed_value(att.get_value_as_string()):  # type: ignore[attr-defined]
                return False
        elif isinstance(att, SimpleAttribute):
            if att.attribute_desc is not self:
                return False
        return True

    def delete(self) -> None:
        self.delete_all_fcts()
        self._owner.remove_attribute_desc(self._name)

    def update_amalgamation_fcts(self, c: Concept, active_sim: object) -> None:
        for f in c.get_available_amalgam_fcts():
            f.set_active(self, True)
            f.set_active_fct(self, active_sim)
            f.set_weight(self.name, 1)
        for sub in c.sub_concepts.values():
            self.update_amalgamation_fcts(sub, active_sim)

    def get_representation(self) -> dict[str, Any]:
        ret: dict[str, Any] = {}
        ret["name"] = self._name
        ret["type"] = self.__class__.__name__
        ret["owner"] = self._owner
        ret["is_multiple"] = self._is_multiple
        return ret

    def __str__(self) -> str:
        return self._name
