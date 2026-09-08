from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.similarity.similarity import Similarity


class RetrievalResult:

    def __init__(self, retrieval_id: str | None, result: list[tuple[Instance, Similarity]]):
        self.retrieval_id = retrieval_id
        self.result = result
