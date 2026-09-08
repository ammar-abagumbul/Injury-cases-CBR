from __future__ import annotations

from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.retrieval.retrieval_engine import RetrievalEngine
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.instance import Instance
    from hklii_psla.cbr.core.i_case_base import ICaseBase
    from hklii_psla.cbr.core.project import Project


class SequentialRetrieval(RetrievalEngine):
    """Computes the similarity of the query against every case in the case
    base. Cases may belong to a different concept than the query; in that
    case the project's inheritance taxonomy scales the local similarity."""

    def __init__(self, prj: Project):
        self.prj = prj

    @override
    def retrieve(self, cb: ICaseBase, q: Instance) -> list[tuple[Instance, Similarity]]:
        result: list[tuple[Instance, Similarity]] = []
        for c in cb.get_cases():
            self.set_current_case(c)
            q_id = q.concept.name
            c_id = c.concept.name

            inheritance_sim = Similarity.get(1.0)
            if q_id != c_id:
                inheritance_sim = self.prj.get_inh_fct().calculate_similarity_by_name(q_id, c_id)

            inh_desc = self.prj.get_inh_fct().desc
            q_att = inh_desc.get_attribute(q_id)
            c_att = inh_desc.get_attribute(c_id)
            common_ancestor = self.prj.get_inh_fct().get_common_ancestor(q_att, c_att)  # type: ignore[arg-type]

            local_sim = Similarity.get(0.0)
            from hklii_psla.cbr.core.casebase.symbol_attribute import SymbolAttribute

            if isinstance(common_ancestor, SymbolAttribute):
                local_sim = self._calculate_local_sim(common_ancestor, q, c)

            sim = Similarity.get(local_sim.value * inheritance_sim.value)
            result.append((c, sim))
        return result

    def _calculate_local_sim(self, first, q: Instance, c: Instance) -> Similarity:
        concept = self.prj.get_concept_by_id(first.value)
        if concept is None:
            return Similarity.INVALID_SIM
        amalgam = concept.get_active_amalgam_fct()
        if amalgam is None:
            return Similarity.INVALID_SIM
        return amalgam.calculate_similarity(q, c)

    @override
    def retrieve_k(self, cb: ICaseBase, q: Instance, k: int) -> list[tuple[Instance, Similarity]]:
        cases = list(cb.get_cases())
        number_of_cases = min(k, len(cases))

        result: list[tuple[Instance, Similarity]] = []
        lowest_sim = Similarity.get(1.00)

        it = iter(cases)
        for counter, case in enumerate(it, start=1):
            self.set_current_case(case)
            amalgam = q.concept.get_active_amalgam_fct()
            current_sim = amalgam.calculate_similarity(q, self.get_current_case())
            if current_sim.value < lowest_sim.value:
                lowest_sim = current_sim
            self._add_sorted(result, (self.get_current_case(), current_sim))
            if counter == number_of_cases:
                break

        for case in it:
            self.set_current_case(case)
            amalgam = q.concept.get_active_amalgam_fct()
            current_sim = amalgam.calculate_similarity(q, self.get_current_case())
            if current_sim.value > lowest_sim.value:
                result.pop()
                self._add_sorted(result, (self.get_current_case(), current_sim))
                lowest_sim = min(result, key=lambda p: p[1].value)[1]

        return result

    def _add_sorted(self, lst: list[tuple[Instance, Similarity]], pair: tuple[Instance, Similarity]) -> None:
        index = 0
        for current in lst:
            if pair[1].value >= current[1].value:
                break
            index += 1
        lst.insert(index, pair)

    @override
    def retrieve_k_sorted(self, cb: ICaseBase, q: Instance, k: int) -> list[tuple[Instance, Similarity]]:
        return self.retrieve_k(cb, q, k)

    @override
    def retrieve_sorted(self, cb: ICaseBase, q: Instance) -> list[tuple[Instance, Similarity]]:
        result = self.retrieve(cb, q)
        result.sort(key=lambda p: p[1].value, reverse=True)
        return result
