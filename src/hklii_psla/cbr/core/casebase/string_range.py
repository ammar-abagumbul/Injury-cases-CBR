from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.casebase.range import Range
from hklii_psla.cbr.core.casebase.string_attribute import StringAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.string_desc import StringDesc
    from hklii_psla.cbr.core.project import Project


class StringRange(Range):
    """StringAttribute objects are not referenced/cached; each call creates a new one."""

    def __init__(self, prj: Project, desc: StringDesc):
        super().__init__(prj)
        self._desc = desc

    def get_string_value(self, s: str) -> StringAttribute:
        return StringAttribute(self._desc, s)

    @override
    def get_attribute(self, obj: object) -> Attribute | None:
        if isinstance(obj, str):
            assert self.project is not None
            if self.project.is_special_attribute(obj):
                return self.project.get_special_attribute(obj)
            return self.get_string_value(obj)
        return None

    @override
    def parse_value(self, string: str):
        return self.get_string_value(string)
