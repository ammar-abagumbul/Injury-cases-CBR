from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.i_case_base import ICaseBase

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.project import Project


class DefaultCaseBase(ICaseBase):
    """Represents cases using myCBR's default in-memory attribute/instance model."""

    def __init__(self, project: Project, name: str):
        if not name or not name.strip():
            raise ValueError("Cannot add case base with empty name")
        if project.get_cb(name) is not None:
            raise ValueError(f'Project already has a case base with name "{name}"')
        self._cases: dict[str, Instance] = {}
        self._name = name
        self._project = project
        self.author: str | None = None
        self.date = None

    @property
    @override
    def name(self) -> str:
        return self._name

    @override
    def set_name(self, name: str) -> None:
        if not name or not name.strip():
            raise ValueError("Cannot give a case base an empty name!")
        if self._project.get_cb(name) is not None:
            raise ValueError(
                f'Case base with name "{self._name}" already exists in project "{self._project.name}"'
            )
        self._name = name

    def get_cases(self) -> list[Instance]:
        return list(self._cases.values())

    def add_case(self, case: Instance) -> None:
        if self._cases.get(case.name) is None:
            self._cases[case.name] = case

    def remove_case(self, name: str) -> bool:
        return self._cases.pop(name, None) is not None

    @override
    def contains_case(self, name: str) -> Instance | None:
        return self._cases.get(name)

    @property
    @override
    def project(self) -> Project:
        return self._project
