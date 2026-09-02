from hklii_psla.cbr.core.model.concept import Concept


class Project(Concept):
    """Root concept of a project's concept hierarchy.

    A Project is the top of the hierarchy, so it has no super-concept
    and no parent project. Passing ``None`` for both breaks the
    Concept <-> Project bootstrap cycle.
    """

    def __init__(self, ID: str):
        super().__init__(ID=ID, project=None, super_concept=None)

    def has_concept_with_id(self, id: str) -> bool:
        return bool(self.all_sub_concepts.get(id))
