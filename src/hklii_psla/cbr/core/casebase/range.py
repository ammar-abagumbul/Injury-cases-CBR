from __future__ import annotations

from abc import ABC, abstractmethod

from typing import final

from hklii_psla.cbr.core.casebase.attribute import Attribute
from hklii_psla.cbr.core.project import Project


class Range(ABC):

    def __init__(self, project: Project | None):
        self._project = project

    @property
    @abstractmethod
    def attribute(self) -> Attribute:
        ...

    @abstractmethod
    def parse_value(self, string: str):
        ...

    @property
    def project(self) -> Project | None:
        return self._project

    @final
    @project.setter
    def set_project(self, value: Project | None):
        self._project = value
