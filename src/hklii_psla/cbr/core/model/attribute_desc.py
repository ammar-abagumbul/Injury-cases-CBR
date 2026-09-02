from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any

from hklii_psla.cbr.core.casebase.simple_attribute import SimpleAttribute
from hklii_psla.cbr.core.model.concept import Concept


class AttributeDesc(ABC):

    DELETE_NOTIFICATION = "delete_notification"

    def __init__(self, owner: Concept, name: str):
        self._name = name
        self._owner = owner
        self._is_solution: bool = False
        self._is_multiple: bool = False

    @property
    def name(self):
        return self._name

    @abstractmethod
    def can_override(self, desc: AttributeDesc) -> bool:
        ...

    def get_representation(self) -> dict[str, Any]:
        ret: dict[str, Any] = {}
        ret["name"] = self._name
        ret["type"] = self.__class__.__name__
        ret["owner"] = self._owner
        ret["is_multiple"] = self._is_multiple
        return ret
