from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.i_case_base import ICaseBase
    from hklii_psla.cbr.core.similarity.similarity import Similarity


class RetrievalEngine(ABC):
    """Retrieves the most similar cases to a query from a case base.
    Implements the strategy pattern; more efficient engines than
    SequentialRetrieval can be added for retrieving only the top-k."""

    current_case: Instance | None = None

    @abstractmethod
    def retrieve(self, cb: ICaseBase, q: Instance) -> list[tuple[Instance, Similarity]]:
        ...

    @abstractmethod
    def retrieve_sorted(self, cb: ICaseBase, q: Instance) -> list[tuple[Instance, Similarity]]:
        ...

    @abstractmethod
    def retrieve_k(self, cb: ICaseBase, q: Instance, k: int) -> list[tuple[Instance, Similarity]]:
        ...

    @abstractmethod
    def retrieve_k_sorted(self, cb: ICaseBase, q: Instance, k: int) -> list[tuple[Instance, Similarity]]:
        ...

    def set_current_case(self, c: Instance) -> None:
        self.current_case = c

    def get_current_case(self) -> Instance | None:
        return self.current_case
