from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hklii_psla.cbr.core.similarity.config.multiple_config import MultipleConfig
from hklii_psla.cbr.core.similarity.sim_fct import SimFct
from hklii_psla.cbr.core.similarity.similarity import Similarity
from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.project import Project

DEFAULT_ICD_PATH = Path(__file__).resolve().parents[4] / "data" / "ICD-11.json"


def code_segments(code: str) -> list[str]:
    """Split an ICD code into its hierarchy segments.

    The code concatenates the path but not with a fixed separator::

        NA00     -> ["NA00"]              block
        NA00.0   -> ["NA00", ".0"]        category   (appends ".0")
        NA00.01  -> ["NA00", ".0", "1"]   leaf       (appends "1", no dot)
        NA00.0Y  -> ["NA00", ".0", "Y"]
    """
    if "." not in code:
        return [code]
    base, rest = code.split(".", 1)
    return [base, "." + rest[0]] + list(rest[1:])


class IcdTaxonomy:
    """``code -> ancestor path`` index over ``data/ICD-11.json``.

    ``path`` is the sequence of node codes from the section down to the node,
    so the longest shared prefix of two paths identifies their LCA.
    """

    def __init__(self, tree: list[dict[str, Any]]):
        self._path: dict[str, tuple[str, ...]] = {}
        self._description: dict[str, str] = {}
        self._section: dict[str, str] = {}
        self._block_section: dict[str, str] = {}
        self._build(tree)

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_ICD_PATH) -> IcdTaxonomy:
        with Path(path).open() as f:
            return cls(json.load(f))

    def _build(self, tree: list[dict[str, Any]]) -> None:
        for section in tree:
            title = section.get("section") or section.get("description", "")
            token = f"{title}"

            def walk(node: dict[str, Any], ancestors: tuple[str, ...],
                     token: str = token) -> None:
                code = node.get("code")
                if code is not None:
                    if not ancestors:
                        self._block_section[code] = token
                    self._path.setdefault(code, (token,) + ancestors + (code,))
                    self._description.setdefault(code, node.get("description", ""))
                    self._section.setdefault(code, title)
                for child in node.get("children", []):
                    walk(child, ancestors + ((code,) if code is not None else ()))

            walk(section, ())

    def __len__(self) -> int:
        return len(self._path)

    def is_known(self, code: str) -> bool:
        return code in self._path

    def path_of(self, code: str) -> tuple[str, ...] | None:
        return self._path.get(code)

    def describe(self, code: str) -> str:
        return self._description.get(code, "")

    def section_of(self, code: str) -> str | None:
        return self._section.get(code)

    def lca_path(self, code_a: str, code_b: str) -> tuple[str, ...]:
        pa, pb = self._path.get(code_a, ()), self._path.get(code_b, ())
        common: list[str] = []
        for x, y in zip(pa, pb):
            if x != y:
                break
            common.append(x)
        return tuple(common)

    def similarity(self, code_a: str, code_b: str) -> float:
        """Normalised LCA depth in ``[0.0, 1.0]``.

        Uses the paths built from the JSON tree (authoritative).  Raises
        ``KeyError`` if either code is unknown; use ``is_known`` to guard.
        """
        pa = self._path[code_a]
        pb = self._path[code_b]
        common = 0
        for x, y in zip(pa, pb):
            if x != y:
                break
            common += 1
        return common / max(len(pa), len(pb), 1)


class IcdCodeSimFct(SimFct):
    """Local similarity function over ICD-11 codes.

    Unlike ``SymbolFct``/``TaxonomyFct`` it stores no ``n x n`` table: the
    metric is computed from the taxonomy paths in O(depth).  Add it to a
    ``SymbolDesc`` with ``desc.add_function(fct, active=True)``.

    ``fallback`` is returned when a value is unknown to the taxonomy (e.g. a
    section-level stop in ``icd_classifier``, or a missing code).
    """

    def __init__(
        self,
        project: Project,
        desc: AttributeDesc,
        taxonomy: IcdTaxonomy,
        name: str = "icd_taxonomy_sim",
        fallback: float = 0.0,
    ):
        super().__init__(project, desc, name)
        self.taxonomy = taxonomy
        self.fallback = fallback
        self.mc = MultipleConfig.DEFAULT_CONFIG
        self.is_symmetric = True

    @staticmethod
    def _value(att: Attribute | None) -> str | None:
        if att is None:
            return None
        return att.get_value_as_string()  # type: ignore[attr-defined]

    def calculate_similarity(self, a1: Attribute | None, a2: Attribute | None) -> Similarity:

        if isinstance(a1, SpecialAttribute) or isinstance(a2, SpecialAttribute):
            return self.prj.calculate_special_similarity(a1, a2)
        if isinstance(a1, MultipleAttribute) and isinstance(a2, MultipleAttribute):
            return self.prj.calculate_multiple_attribute_similarity(self, a1, a2)

        v1, v2 = self._value(a1), self._value(a2)
        if v1 is None or v2 is None:
            return Similarity.get(self.fallback)
        if v1 == v2:
            return Similarity.get(1.0)
        if not (self.taxonomy.is_known(v1) and self.taxonomy.is_known(v2)):
            return Similarity.get(self.fallback)
        return Similarity.get(self.taxonomy.similarity(v1, v2))

    def get_multiple_config(self) -> MultipleConfig:
        return self.mc

    def set_multiple_config(self, mc: MultipleConfig) -> None:
        self.mc = mc

    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.symbol_desc import SymbolDesc

        if isinstance(new_desc, SymbolDesc):
            f = IcdCodeSimFct(self.prj, new_desc, self.taxonomy, self.name, self.fallback)
            new_desc.add_function(f, is_active)
