from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.attribute import Attribute
from hklii_psla.cbr.core.casebase.range import Range

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.project import Project


class ConceptRange(Range):

    def __init__(self, project: Project | None, concept: Concept | None):
        super().__init__(project)
        self._concept = concept
        self._instances: dict[str, Instance] = {}

    @property
    def concept(self) -> Concept | None:
        return self._concept

    def get_instance(self, name: str) -> Instance:
        from hklii_psla.cbr.core.casebase.instance import Instance

        att = self._instances.get(name)
        if att is None:
            assert self._concept is not None
            att = Instance(self._concept, name)
        return att

    def add(self, instance: Instance) -> bool:
        if self._instances.get(instance.name) is None:
            self._instances[instance.name] = instance
            return True
        return False

    def contains(self, name: str) -> Instance | None:
        return self._instances.get(name)

    @property
    def instances(self):
        return self._instances.values()

    @override
    def get_attribute(self, obj: object) -> Attribute | None:
        if isinstance(obj, str):
            assert self._project is not None
            if self._project.is_special_attribute(obj):
                return self._project.get_special_attribute(obj)
            assert self._concept is not None
            instance = self._concept.get_instance(obj)
            if instance is None:
                instance = self.get_instance(obj)
            return instance
        return None

    def rename_instance(self, name: str, new_name: str) -> None:
        instance = self._instances.pop(name, None)
        if instance is not None:
            self._instances[new_name] = instance

    @override
    def parse_value(self, string: str):
        return self.get_instance(string)

    def set_all_instances_single(self, desc: AttributeDesc) -> None:
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        for instance in self._instances.values():
            att = instance.get_att_for_desc(desc)
            if isinstance(att, MultipleAttribute):
                values = att.values
                assert self._project is not None
                new_att = values[0] if values else self._project.get_special_attribute(
                    self._project.UNDEFINED_SPECIAL_ATTRIBUTE
                )
                instance.add_attribute(desc, new_att)

    def set_all_instances_multiple(self, desc: AttributeDesc) -> None:
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute

        for instance in self._instances.values():
            att = instance.get_att_for_desc(desc)
            assert self._project is not None
            if att is not None and not self._project.is_special_attribute(att.get_value_as_string()):
                instance.add_attribute(desc, MultipleAttribute(desc, [att]))

    def clear(self) -> None:
        self._instances.clear()

    def remove(self, name: str) -> None:
        self._instances.pop(name, None)
        if self._project is not None:
            self._project.remove_case(name)
