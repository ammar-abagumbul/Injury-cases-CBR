"""Coarse (non-leaf) ICD-11 grouping.

Reuses ``experiment4``'s ``IcdTaxonomy`` rather than re-parsing ``data/ICD-11.json`` a second
time. Where ``IcdTaxonomy`` computes similarity between two full codes,
this module answers a different question: given one code, what's its
ancestor at a fixed, shallow depth in the tree?

Per the ICD-11 browser hierarchy (chapter -> section -> block -> category
-> subcategory), a code's ``path_of()`` looks like::

    ("Injuries to the head", "NA00", "NA00.0", "NA00.01")
     ^ section title (no code)  ^block  ^category  ^leaf

``truncate_code(taxonomy, "NA00.01", max_depth=2)`` returns ``"NA00.0"``
(the category, index 2) rather than descending to the full leaf — matching
the intended stopping depth ("no need to expand it further").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hklii_psla.experiments.experiment4.icd_similarity import IcdTaxonomy
from hklii_psla.severity.models import IcdGroup

if TYPE_CHECKING:
    from hklii_psla.schemas import Injury

DEFAULT_MAX_DEPTH = 2  # 1=block (e.g. "NA00"), 2=category (e.g. "NA00.0")


def truncate_code(
    taxonomy: IcdTaxonomy, code: str | None, max_depth: int = DEFAULT_MAX_DEPTH
) -> str | None:
    """Return the ancestor of ``code`` at ``max_depth`` code-levels deep.

    Depth is counted in code-bearing tree levels only (the section title
    isn't a code, so it's excluded): depth 1 = block, depth 2 = category,
    etc. If ``code`` is unknown to the taxonomy, it's returned unchanged
    (nothing to truncate). If ``code`` is already shallower than
    ``max_depth`` (e.g. a section-level stop from ``icd_classifier``), the
    deepest available ancestor is returned as-is.
    """
    if code is None:
        return None
    path = taxonomy.path_of(code)
    if path is None:
        return code
    # path[0] is the section title (not a code); real codes start at path[1].
    index = min(max_depth, len(path) - 1)
    return path[index]


def group_injuries(
    injuries: list[Injury], taxonomy: IcdTaxonomy, max_depth: int = DEFAULT_MAX_DEPTH
) -> list[IcdGroup]:
    """Map each injury's ICD code to its coarse group, in injury order."""
    groups: list[IcdGroup] = []
    for inj in injuries:
        group_code = truncate_code(taxonomy, inj.icd_code, max_depth)
        groups.append(IcdGroup(
            injury_id=inj.injury_id,
            original_code=inj.icd_code,
            group_code=group_code,
            group_description=taxonomy.describe(group_code) if group_code else None,
            truncated=bool(group_code and inj.icd_code and group_code != inj.icd_code),
        ))
    return groups
