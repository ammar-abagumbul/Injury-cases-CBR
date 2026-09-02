from __future__ import annotations

from typing import TYPE_CHECKING, final

from hklii_psla.cbr.core.casebase.range import Range

if TYPE_CHECKING:
    from hklii_psla.cbr.core.model.concept import Concept
    from hklii_psla.cbr.core.project import Project


@final
class ConceptRange(Range):
    _concept: Concept | None
    _project: Project | None

    def __init__(self, project: Project | None, concept: Concept | None):
        super().__init__(project)
        self._project = project
        self._concept = concept
